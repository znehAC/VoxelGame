using Godot;
using System.Threading;

public enum ChunkState
{
    Idle,               // Not loaded
    AwaitingData,       // Loaded, waiting for a data generation job
    GeneratingData,     // A worker thread is currently generating its voxel data
    AwaitingMesh,       // Data is ready, but it's waiting for neighbors to be ready before meshing
    Meshing,            // A worker thread is currently generating its mesh
    Ready               // Mesh is visible and has collision
}

public partial class Chunk : GodotObject
{
    public const int Size = 32;
    public Vector3I Position { get; private set; }
    public int Lod { get; set; } = 0;

    public bool IsFullyOpaque { get; private set; } = false;
    public bool IsEmpty { get; private set; } = false;

    public ChunkState State { get; set; } = ChunkState.AwaitingData;
    
    public double GhostLifeTime { get; set; } = 0.0;

    private readonly World _world;
    private readonly byte[] _voxels = new byte[Size * Size * Size];

    // Persistant MeshInstance3D to prevent tree churn
    private MeshInstance3D _meshInstance;
    public MeshInstance3D MeshInstance => _meshInstance;
    
    private Rid _bodyRid;
    private Rid _shapeRid;
    private Material _material;

    public CancellationTokenSource JobCancellationTokenSource { get; private set; }

    public Chunk(World world, Vector3I position)
    {
        _world = world;
        Position = position;
        JobCancellationTokenSource = new CancellationTokenSource();
    }

    public byte[] GetVoxels()
    {
        return _voxels;
    }

    public void Reset(Vector3I position)
    {
        // Don't free the MeshInstance, just hide and reset it
        if (GodotObject.IsInstanceValid(_meshInstance))
        {
            _meshInstance.Visible = false;
            _meshInstance.Mesh = null;
        }

        CleanupPhysics(); // Physics is cheap to recreate or could be reused too, but safer to recreate RIDs
        
        Position = position;
        State = ChunkState.AwaitingData;
        IsFullyOpaque = false;
        IsEmpty = false;
        GhostLifeTime = 0.0;
        System.Array.Clear(_voxels, 0, _voxels.Length);

        if (JobCancellationTokenSource.IsCancellationRequested)
        {
            JobCancellationTokenSource.Dispose();
            JobCancellationTokenSource = new CancellationTokenSource();
        }
    }

    public void Initialize(Rid scenario, Rid space, Material material, float globalScale = 1.0f)
    {
        _material = material;
        // Combined scale: LOD Scale (2^LOD) * Global Voxel Scale (e.g. 0.1)
        float combinedScale = (1 << Lod) * globalScale;
        
        // Prepare MeshInstance
        if (!GodotObject.IsInstanceValid(_meshInstance))
        {
            _meshInstance = new MeshInstance3D();
            _meshInstance.Name = "ChunkNode"; 
            _meshInstance.CastShadow = GeometryInstance3D.ShadowCastingSetting.On;
            _world.AddChild(_meshInstance);
        }

        _meshInstance.MaterialOverride = material;
        // Position is VoxelCoordinate * Size * CombinedScale
        _meshInstance.Position = (Vector3)Position * Size * combinedScale;
        _meshInstance.Scale = Vector3.One * combinedScale;
        _meshInstance.Visible = false; // Hidden until mesh applied

        // Physics
        Transform3D transform = new Transform3D(Basis.FromScale(Vector3.One * combinedScale), (Vector3)Position * Size * combinedScale);
        
        _bodyRid = PhysicsServer3D.BodyCreate();
        PhysicsServer3D.BodySetMode(_bodyRid, PhysicsServer3D.BodyMode.Static);
        PhysicsServer3D.BodySetSpace(_bodyRid, space);
        PhysicsServer3D.BodySetState(_bodyRid, PhysicsServer3D.BodyState.Transform, transform);
    }

    public void SetVisible(bool visible)
    {
        if (GodotObject.IsInstanceValid(_meshInstance))
        {
            // Only show if we actually have a mesh
            if (visible && _meshInstance.Mesh == null) return;
            _meshInstance.Visible = visible;
        }
    }

    public void PrepareData(FastNoiseLite continentalnessNoise, FastNoiseLite erosionNoise, FastNoiseLite caveNoise, CancellationToken token)
    {
        if (State != ChunkState.GeneratingData) return;
        GenerateTerrainFromNoise(continentalnessNoise, erosionNoise, caveNoise, token);
        State = ChunkState.AwaitingMesh;
    }

    public void ApplyMeshData(MeshData meshData)
    {
        if (meshData.Vertices != null && meshData.Vertices.Length > 0)
        {
            // Debug Print
            Vector3 visualPos = GodotObject.IsInstanceValid(_meshInstance) ? _meshInstance.GlobalPosition : Vector3.Zero;
            GD.Print($"[Chunk] Applying Mesh: {Position}, Verts: {meshData.Vertices.Length}, VisPos: {visualPos}");
            
            var arrays = new Godot.Collections.Array();
            arrays.Resize((int)Mesh.ArrayType.Max);
            arrays[(int)Mesh.ArrayType.Vertex] = Variant.CreateFrom(meshData.Vertices);
            arrays[(int)Mesh.ArrayType.Normal] = Variant.CreateFrom(meshData.Normals);
            arrays[(int)Mesh.ArrayType.Color] = Variant.CreateFrom(meshData.Colors);
            arrays[(int)Mesh.ArrayType.Index] = Variant.CreateFrom(meshData.Indices);

            var arrayMesh = new ArrayMesh();
            arrayMesh.AddSurfaceFromArrays(Mesh.PrimitiveType.Triangles, arrays);
            
            if (GodotObject.IsInstanceValid(_meshInstance))
            {
                _meshInstance.Mesh = arrayMesh;
                // Note: Position/Scale were set in Initialize
            }
        }
        else
        {
             // Empty mesh
             if (GodotObject.IsInstanceValid(_meshInstance))
             {
                 _meshInstance.Mesh = null;
                 _meshInstance.Visible = false;
             }
        }
    }

    public void ApplyCollisionData(MeshData meshData)
    {
        if (!_bodyRid.IsValid) return;

        if (_shapeRid.IsValid)
        {
            PhysicsServer3D.BodyClearShapes(_bodyRid);
            PhysicsServer3D.FreeRid(_shapeRid);
            _shapeRid = default;
        }

        if (meshData.Indices != null && meshData.Indices.Length > 0)
        {
            var faces = new Vector3[meshData.Indices.Length];
            for (int i = 0; i < meshData.Indices.Length; i++)
            {
                faces[i] = meshData.Vertices[meshData.Indices[i]];
            }

            // Use Resource wrapper to ensure correct data formatting for Godot 4
            var shape = new ConcavePolygonShape3D();
            shape.Data = faces;
            
            // We can take the RID from the resource. 
            // Note: The resource manages the RID. If we let the resource go out of scope, 
            // the RID might be freed if not referenced? 
            // Actually, BodyAddShape doesn't take ownership. 
            // We need to keep the resource alive or manually manage the RID.
            // For safety and simplicity, we'll keep the Resource alive if we could, 
            // BUT our architecture stores _shapeRid. 
            // Let's just use the server to create it, but use the PROPER data format if possible.
            // OR: Just keep the shape resource alive? No, Chunk structure expects RID.
            
            // BETTER: Use the Resource to generate the RID, then hack it? No.
            // The error !d.has("faces") implies Godot expects a Dictionary for ConcavePolygonShape.
            
            var dictionary = new Godot.Collections.Dictionary();
            dictionary["faces"] = faces;
            
            _shapeRid = PhysicsServer3D.ConcavePolygonShapeCreate();
            PhysicsServer3D.ShapeSetData(_shapeRid, dictionary);
            PhysicsServer3D.BodyAddShape(_bodyRid, _shapeRid);
        }
    }

    public void Unload()
    {
        JobCancellationTokenSource.Cancel();
        
        // On strict unload, we might want to free the node to free memory
        // But ChunkPool recycles the Chunk object.
        // We keep the MeshInstance alive for the next use of this Chunk object.
        if (GodotObject.IsInstanceValid(_meshInstance))
        {
            _meshInstance.Visible = false;
            _meshInstance.Mesh = null;
        }
        CleanupPhysics();
    }

    private void CleanupPhysics()
    {
        if (_bodyRid.IsValid) PhysicsServer3D.FreeRid(_bodyRid);
        if (_shapeRid.IsValid) PhysicsServer3D.FreeRid(_shapeRid);
        _bodyRid = default;
        _shapeRid = default;
    }

    public byte GetVoxelInternal(int x, int y, int z)
    {
        if (x < 0 || y < 0 || z < 0 || x >= Size || y >= Size || z >= Size) return VoxelTypes.Air;
        return _voxels[(z * Size * Size) + (y * Size) + x];
    }

    public byte GetVoxel(int x, int y, int z)
    {
        if (x >= 0 && x < Size && y >= 0 && y < Size && z >= 0 && z < Size)
        {
            return GetVoxelInternal(x, y, z);
        }

        var worldPos = new Vector3I(Position.X * Size + x, Position.Y * Size + y, Position.Z * Size + z);
        var neighborChunkPos = new Vector3I(
            (int)Mathf.Floor(worldPos.X / (float)Size),
            (int)Mathf.Floor(worldPos.Y / (float)Size),
            (int)Mathf.Floor(worldPos.Z / (float)Size));

        var neighborKey = new Vector4I(neighborChunkPos.X, neighborChunkPos.Y, neighborChunkPos.Z, Lod);

        if (_world.VoxelChunks.TryGetValue(neighborKey, out Chunk neighborChunk))
        {
            var localPos = new Vector3I(
                (worldPos.X % Size + Size) % Size,
                (worldPos.Y % Size + Size) % Size,
                (worldPos.Z % Size + Size) % Size);
            return neighborChunk.GetVoxelInternal(localPos.X, localPos.Y, localPos.Z);
        }
        return VoxelTypes.Air;
    }

    public void SetVoxel(int x, int y, int z, byte value)
    {
        _voxels[(z * Size * Size) + (y * Size) + x] = value;
    }

    public void SetVoxels(byte[] newVoxels, bool? isEmpty = null)
    {
        if (newVoxels.Length != _voxels.Length)
        {
            GD.PrintErr($"Chunk {Position}: SetVoxels length mismatch!");
            return;
        }
        System.Array.Copy(newVoxels, _voxels, _voxels.Length);
        
        if (isEmpty.HasValue)
        {
            IsEmpty = isEmpty.Value;
            if (IsEmpty)
            {
                IsFullyOpaque = false;
                // GD.Print($"[CHUNK EMPTY] Pos: {Position} LOD: {Lod}");
            }
            else
            {
                IsFullyOpaque = CheckOpacity();
            }
        }
        else
        {
            UpdateProperties();
        }
    }

    private void UpdateProperties()
    {
        bool hasAir = false;
        bool hasSolid = false;

        for (int i = 0; i < _voxels.Length; i++)
        {
            if (_voxels[i] == VoxelTypes.Air) hasAir = true;
            else hasSolid = true;

            if (hasAir && hasSolid) break; 
        }

        IsFullyOpaque = !hasAir;
        IsEmpty = !hasSolid;
    }

    private bool CheckOpacity()
    {
        for (int i = 0; i < _voxels.Length; i++)
        {
            if (_voxels[i] == VoxelTypes.Air) return false;
        }
        return true;
    }

    public void GenerateTerrainFromNoise(FastNoiseLite continentalnessNoise, FastNoiseLite erosionNoise, FastNoiseLite caveNoise, CancellationToken token)
    {
        bool containsAir = false;
        const int seaLevel = 50;
        const int mountainLevel = 100;

        for (int x = 0; x < Size; x++)
        {
            for (int z = 0; z < Size; z++)
            {
                token.ThrowIfCancellationRequested();
                int worldVoxelX = Position.X * Size + x;
                int worldVoxelZ = Position.Z * Size + z;

                float continentalValue = continentalnessNoise.GetNoise2D(worldVoxelX, worldVoxelZ);
                float erosionValue = erosionNoise.GetNoise2D(worldVoxelX, worldVoxelZ);

                int surfaceHeight = seaLevel;
                if (continentalValue > 0)
                {
                    surfaceHeight += (int)(continentalValue * 40) + (int)(erosionValue * 10);
                }

                for (int y = 0; y < Size; y++)
                {
                    int worldVoxelY = Position.Y * Size + y;

                    float caveNoiseValue = caveNoise.GetNoise3D(worldVoxelX, worldVoxelY, worldVoxelZ);
                    if (caveNoiseValue > 0.7f)
                    {
                        SetVoxel(x, y, z, VoxelTypes.Air);
                        containsAir = true;
                        continue;
                    }

                    if (worldVoxelY > surfaceHeight)
                    {
                        SetVoxel(x, y, z, VoxelTypes.Air);
                        containsAir = true;
                    }
                    else if (worldVoxelY == surfaceHeight)
                    {
                        if (surfaceHeight > mountainLevel)
                        {
                            SetVoxel(x, y, z, VoxelTypes.Stone);
                        }
                        else
                        {
                            SetVoxel(x, y, z, VoxelTypes.Grass);
                        }
                    }
                    else if (worldVoxelY > surfaceHeight - 5)
                    {
                        if (surfaceHeight > mountainLevel)
                        {
                            SetVoxel(x, y, z, VoxelTypes.Stone);
                        }
                        else
                        {
                            SetVoxel(x, y, z, VoxelTypes.Dirt);
                        }
                    }
                    else
                    {
                        SetVoxel(x, y, z, VoxelTypes.Stone);
                    }
                }
            }
        }

        IsFullyOpaque = !containsAir;
    }
}
