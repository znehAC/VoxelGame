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
    private const int PaddedSize = Size + 2;
    public Vector3I Position { get; private set; }

    public bool IsFullyOpaque { get; private set; } = false;

    public ChunkState State { get; set; } = ChunkState.AwaitingData;

    private readonly World _world;
    private readonly byte[] _voxels = new byte[Size * Size * Size];

    private MeshInstance3D _meshInstance;
    private StaticBody3D _staticBody;
    private CollisionShape3D _collisionShape;

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
        GD.Print($"Resetting chunk {Position} to {position}");
        Position = position;
        State = ChunkState.AwaitingData;
        IsFullyOpaque = false;
        System.Array.Clear(_voxels, 0, _voxels.Length);

        if (JobCancellationTokenSource.IsCancellationRequested)
        {
            JobCancellationTokenSource.Dispose();
            JobCancellationTokenSource = new CancellationTokenSource();
        }
    }

    public void InitializeMeshNode(Node parent, Material material)
    {
        Vector3 chunkWorldOrigin = (Vector3)Position * Size;

        _meshInstance = new MeshInstance3D
        {
            Name = $"ChunkMesh_{Position}",
            MaterialOverride = material,
            GlobalPosition = chunkWorldOrigin
        };

        parent.CallDeferred("add_child", _meshInstance);
    }

    public void PrepareData(FastNoiseLite continentalnessNoise, FastNoiseLite erosionNoise, FastNoiseLite caveNoise, CancellationToken token)
    {
        if (State != ChunkState.GeneratingData) return;

        GenerateTerrainFromNoise(continentalnessNoise, erosionNoise, caveNoise, token);
        State = ChunkState.AwaitingMesh;
    }

    public void ApplyMeshData(MeshData meshData)
    {
        if (!IsInstanceValid(_meshInstance)) return;
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
        if (!IsInstanceValid(_collisionShape)) return;
        var shape = new ConcavePolygonShape3D();
        if (meshData.Indices.Count > 0)
        {
            var faces = new Vector3[meshData.Indices.Count];
            for (int i = 0; i < meshData.Indices.Count; i++)
            {
                faces[i] = meshData.Vertices[meshData.Indices[i]];
            }
            shape.SetFaces(faces);
        }
        _collisionShape.Shape = shape;
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

        if (_world.VoxelChunks.TryGetValue(neighborChunkPos, out Chunk neighborChunk))
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

    public void Unload()
    {
        GD.Print($"Cancelling jobs for chunk {Position}");
        JobCancellationTokenSource.Cancel();

        if (IsInstanceValid(_meshInstance)) _meshInstance.QueueFree();
        if (IsInstanceValid(_staticBody)) _staticBody.QueueFree();
    }
}
