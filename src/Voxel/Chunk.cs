using Godot;

public partial class Chunk : GodotObject
{
    public const int Size = 32;
    private const int PaddedSize = Size + 2;
    public Vector3I Position { get; private set; }

    private bool _isDataGenerated = false;
    public bool IsDataGenerated => _isDataGenerated;

    private readonly World _world;
    private readonly byte[] _voxels = new byte[Size * Size * Size];

    private MeshInstance3D _meshInstance;

    public Chunk(World world, Vector3I position)
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

    public MeshData GenerateMesh()
    {
        byte[] paddedVoxels = BuildPaddedVoxels();
        return GreedyMesher.GenerateMeshData(paddedVoxels, Size, VoxelTypes.Definitions);
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

    public byte GetVoxel(int x, int y, int z)
    {
        if (x >= 0 && x < Size && y >= 0 && y < Size && z >= 0 && z < Size)
        {
            return _voxels[(z * Size * Size) + (y * Size) + x];
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
            return neighborChunk.GetVoxel(localPos.X, localPos.Y, localPos.Z);
        }
        return VoxelTypes.Air;
    }

    public void SetVoxel(int x, int y, int z, byte value)
    {
        _voxels[(z * Size * Size) + (y * Size) + x] = value;
    }

    public void PrepareData(FastNoiseLite terrainNoise, FastNoiseLite caveNoise)
    {
        if (_isDataGenerated) return;
        bool loadedFromFile = LoadChunkFromFile();

        if (!loadedFromFile)
        {
            GenerateTerrainFromNoise(terrainNoise, caveNoise);
        }

        _isDataGenerated = true;
    }


    private bool LoadChunkFromFile()
    {

        return false;
    }

    public void GenerateTerrainFromNoise(FastNoiseLite terrainNoise, FastNoiseLite caveNoise)
    {
        const float worldScale = 0.1f;
        for (int x = 0; x < Size; x++)
        {
            for (int z = 0; z < Size; z++)
            {
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
}
