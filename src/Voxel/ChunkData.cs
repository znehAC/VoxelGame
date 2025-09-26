using Godot;
using System.Threading.Tasks;

public partial class ChunkData : GodotObject
{
    public const int Size = 32;
    private const int PaddedSize = Size + 2;
    public Vector3I Position { get; private set; }

    private readonly World _world;
    private readonly byte[] _voxels = new byte[Size * Size * Size];

    private MeshInstance3D _meshInstance;
    private const byte Air = 0;
    private const byte Dirt = 1;
    private const byte Stone = 2;

    public ChunkData(World world, Vector3I position)
    {
        _world = world;
        Position = position;
    }

    public void InitializeMesh(Node parent, Material material)
    {
        _meshInstance = new MeshInstance3D();
        _meshInstance.Transform = new Transform3D(Basis.Identity, Position * Size);
        _meshInstance.MaterialOverride = material;
        parent.AddChild(_meshInstance);
    }

    private bool _isMeshing = false;

    public async Task RebuildMesh()
    {
        if (_isMeshing) return;
        _isMeshing = true;

        try
        {
            // Build padded voxel buffer
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

            // Run greedy mesher off-thread
            MeshData meshData = await Task.Run(() =>
            {
                return GreedyMesher.GenerateMeshData(paddedVoxels, Size, new Color[]
                {
                    new Color(0,0,0,0),                       // Air (unused)
                    new Color(139f/255f, 69f/255f, 19f/255f, 1f),  // Dirt
                    new Color(128f/255f, 128f/255f, 128f/255f, 1f) // Stone
                });
            });

            // Apply result on main thread
            Callable.From(() => ApplyMeshData(meshData)).CallDeferred();
        }
        catch (System.Exception e)
        {
            GD.PrintErr($"Meshing error in chunk {Position}: {e.Message}");
            _isMeshing = false;
        }
    }

    private void ApplyMeshData(MeshData meshData)
    {
        if (_meshInstance != null)
        {
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
        _isMeshing = false;
    }

    public byte GetVoxel(int x, int y, int z)
    {
        if (x >= 0 && x < Size && y >= 0 && y < Size && z >= 0 && z < Size)
        {
            return _voxels[(y * Size * Size) + (z * Size) + x];
        }

        var worldPos = new Vector3I(Position.X * Size + x, Position.Y * Size + y, Position.Z * Size + z);
        var neighborChunkPos = new Vector3I(
            (int)Mathf.Floor(worldPos.X / (float)Size),
            (int)Mathf.Floor(worldPos.Y / (float)Size),
            (int)Mathf.Floor(worldPos.Z / (float)Size));

        if (_world != null && _world.VoxelChunks.TryGetValue(neighborChunkPos, out ChunkData neighborChunk))
        {
            var localPos = new Vector3I(
                (worldPos.X % Size + Size) % Size,
                (worldPos.Y % Size + Size) % Size,
                (worldPos.Z % Size + Size) % Size);
            return neighborChunk.GetVoxel(localPos.X, localPos.Y, localPos.Z);
        }
        return Air;
    }

    public void SetVoxel(int x, int y, int z, byte value)
    {
        _voxels[(y * Size * Size) + (z * Size) + x] = value;
    }

    public void GenerateTerrain(FastNoiseLite noise)
    {
        for (int x = 0; x < Size; x++)
        {
            for (int z = 0; z < Size; z++)
            {
                float noiseValue = noise.GetNoise2D(x + Position.X * Size, z + Position.Z * Size);
                int surfaceHeight = (Size * 2) / 3 + (int)(noiseValue * 5);
                for (int y = 0; y < Size; y++)
                {
                    if (y > surfaceHeight) SetVoxel(x, y, z, Air);
                    else if (y < Size / 3) SetVoxel(x, y, z, Stone);
                    else SetVoxel(x, y, z, Dirt);
                }
            }
        }
    }
}
