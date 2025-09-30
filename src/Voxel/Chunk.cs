using Godot;
using System.Threading;

// datetime/performance import
using System;
public partial class Chunk : GodotObject
{
    public const int Size = 32;
    private const int PaddedSize = Size + 2;
    public Vector3I Position { get; private set; }

    private volatile bool _isDataGenerated = false;
    public bool IsDataGenerated => _isDataGenerated;

    private readonly World _world;
    private readonly byte[] _voxels = new byte[Size * Size * Size];

    private MeshInstance3D _meshInstance;
    private StaticBody3D _staticBody; // Add this
    private CollisionShape3D _collisionShape; // Add this

    public Chunk(World world, Vector3I position)
    {
        _world = world;
        Position = position;
    }

    public void InitializeMesh(Node parent, Material material)
    {
        Vector3 chunkWorldOrigin = (Vector3)Position * Size;

        _meshInstance = new MeshInstance3D();
        _meshInstance.Name = $"ChunkMesh_{Position}";
        _meshInstance.MaterialOverride = material;
        _meshInstance.GlobalPosition = chunkWorldOrigin;

        _staticBody = new StaticBody3D();
        _staticBody.Name = $"ChunkStaticBody_{Position}";
        _staticBody.GlobalPosition = chunkWorldOrigin;

        _collisionShape = new CollisionShape3D();
        _staticBody.AddChild(_collisionShape);

        // Use CallDeferred to safely add nodes to the scene tree
        parent.CallDeferred("add_child", _meshInstance);
        parent.CallDeferred("add_child", _staticBody);
    }

    public MeshData GenerateMesh(CancellationToken token)
    {
        byte[] paddedVoxels = BuildPaddedVoxels();
        return GreedyMesher.GenerateMeshData(paddedVoxels, Size, VoxelTypes.Definitions, token);
    }

    public void ApplyMeshData(MeshData meshData)
    {
        if (!GodotObject.IsInstanceValid(_meshInstance)) return;

        var newMesh = new ArrayMesh();
        if (meshData.Vertices.Count > 0)
        {
            var arrays = new Godot.Collections.Array();
            arrays.Resize((int)Mesh.ArrayType.Max);
            arrays[(int)Mesh.ArrayType.Vertex] = meshData.Vertices.ToArray();
            arrays[(int)Mesh.ArrayType.Normal] = meshData.Normals.ToArray();
            arrays[(int)Mesh.ArrayType.Color] = meshData.Colors.ToArray();
            arrays[(int)Mesh.ArrayType.Index] = meshData.Indices.ToArray();

            newMesh.AddSurfaceFromArrays(Mesh.PrimitiveType.Triangles, arrays);
        }
        _meshInstance.Mesh = newMesh;
    }

    public void ApplyCollisionData(MeshData meshData)
    {
        if (!GodotObject.IsInstanceValid(_collisionShape)) return;

        // Create a new shape to clear any old data.
        var shape = new ConcavePolygonShape3D();

        // Only set faces if there's something to set.
        if (meshData.Indices.Count > 0)
        {
            // This is the crucial fix. We build an array of vertices
            // where every 3 vertices represent one triangle face,
            // using the indices to get the correct order.
            var faces = new Vector3[meshData.Indices.Count];
            for (int i = 0; i < meshData.Indices.Count; i++)
            {
                faces[i] = meshData.Vertices[meshData.Indices[i]];
            }
            shape.SetFaces(faces);
        }

        _collisionShape.Shape = shape;
    }


    private byte[] BuildPaddedVoxels()
    {
        byte[] paddedVoxels = new byte[PaddedSize * PaddedSize * PaddedSize];
        for (int x = 0; x < PaddedSize; x++)
        {
            for (int y = 0; y < PaddedSize; y++)
            {
                for (int z = 0; z < PaddedSize; z++)
                {
                    paddedVoxels[(z * PaddedSize * PaddedSize) + (y * PaddedSize) + x] =
                        GetVoxel(x - 1, y - 1, z - 1);
                }
            }
        }
        return paddedVoxels;
    }

    public byte GetVoxelInternal(int x, int y, int z)
    {
        return _voxels[(z * Size * Size) + (y * Size) + x];
    }

    // Modify the existing GetVoxel method.
    public byte GetVoxel(int x, int y, int z)
    {
        // lets check performance time here
        var start = DateTime.Now;
        if (x >= 0 && x < Size && y >= 0 && y < Size && z >= 0 && z < Size)
        {
            return GetVoxelInternal(x, y, z);
        }

        var worldPos = new Vector3I(Position.X * Size + x, Position.Y * Size + y, Position.Z * Size + z);
        var neighborChunkPos = new Vector3I(
            (int)Mathf.Floor(worldPos.X / (float)Size),
            (int)Mathf.Floor(worldPos.Y / (float)Size),
            (int)Mathf.Floor(worldPos.Z / (float)Size));

        if (_world != null && _world.VoxelChunks.TryGetValue(neighborChunkPos, out Chunk neighborChunk))
        {
            var localPos = new Vector3I(
                (worldPos.X % Size + Size) % Size,
                (worldPos.Y % Size + Size) % Size,
                (worldPos.Z % Size + Size) % Size);

            // This is the key: call the dumb getter on the neighbor.
            return neighborChunk.GetVoxelInternal(localPos.X, localPos.Y, localPos.Z);
        }

        return VoxelTypes.Air;
    }

    public void SetVoxel(int x, int y, int z, byte value)
    {
        _voxels[(z * Size * Size) + (y * Size) + x] = value;
    }

    public void PrepareData(FastNoiseLite terrainNoise, FastNoiseLite caveNoise, CancellationToken token)
    {
        if (_isDataGenerated) return;
        bool loadedFromFile = LoadChunkFromFile();

        if (!loadedFromFile)
        {
            GenerateTerrainFromNoise(terrainNoise, caveNoise, token);
        }

        _isDataGenerated = true;
    }


    private bool LoadChunkFromFile()
    {
        return false;
    }

    public void GenerateTerrainFromNoise(FastNoiseLite terrainNoise, FastNoiseLite caveNoise, CancellationToken token)
    {
        const float worldScale = 0.1f;
        for (int x = 0; x < Size; x++)
        {
            for (int z = 0; z < Size; z++)
            {
                token.ThrowIfCancellationRequested();
                int worldVoxelX = Position.X * Size + x;
                int worldVoxelZ = Position.Z * Size + z;
                float scaledX = worldVoxelX * worldScale;
                float scaledZ = worldVoxelZ * worldScale;

                const int seaLevel = 0;

                const int terrainAmplitude = 200;

                float terrainValue = terrainNoise.GetNoise2D(scaledX, scaledZ);
                int surfaceHeight = seaLevel + (int)(terrainValue * terrainAmplitude);

                for (int y = 0; y < Size; y++)
                {
                    int worldVoxelY = Position.Y * Size + y;
                    float scaledY = worldVoxelY * worldScale;

                    float caveValue = caveNoise.GetNoise3D(scaledX, scaledY, scaledZ);
                    if (caveValue > 0.6f)
                    {
                        SetVoxel(x, y, z, VoxelTypes.Air);
                        continue;
                    }


                    byte voxelType;
                    if (worldVoxelY > surfaceHeight)
                    {
                        if (worldVoxelY <= seaLevel)
                        {
                            voxelType = VoxelTypes.Water;
                        }
                        else
                        {
                            voxelType = VoxelTypes.Air;
                        }
                    }
                    else if (worldVoxelY == surfaceHeight)
                    {
                        voxelType = VoxelTypes.Grass;
                    }
                    else if (worldVoxelY > surfaceHeight - 40)
                    {
                        voxelType = VoxelTypes.Dirt;
                    }
                    else
                    {
                        voxelType = VoxelTypes.Stone;
                    }

                    SetVoxel(x, y, z, voxelType);
                }
            }
        }
    }

    public void Unload()
    {
        if (GodotObject.IsInstanceValid(_meshInstance))
            _meshInstance.QueueFree();
        if (GodotObject.IsInstanceValid(_staticBody))
            _staticBody.QueueFree();
    }
}
