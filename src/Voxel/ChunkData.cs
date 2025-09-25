using Godot;

public partial class ChunkData : GodotObject
{
    public const int Size = 32;
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
        this.Position = position;
    }

    public void InitializeMesh(Node parent, Material material)
    {
        _meshInstance = new MeshInstance3D();
        _meshInstance.Position = this.Position * Size;
        _meshInstance.MaterialOverride = material;
        parent.AddChild(_meshInstance);
    }

    public void RebuildMesh()
    {
        if (_meshInstance == null) return;
        _meshInstance.Mesh = GenerateMesh();
    }


    private ArrayMesh GenerateMesh()
    {
        var st = new SurfaceTool();
        st.Begin(Mesh.PrimitiveType.Triangles);

        var axes = new[] { Vector3I.Axis.X, Vector3I.Axis.Y, Vector3I.Axis.Z };

        foreach (var axis in axes)
        {
            int uAxis = ((int)axis + 1) % 3;
            int vAxis = ((int)axis + 2) % 3;

            var x = new int[3];
            var dir = new Vector3I();
            dir[(int)axis] = 1;

            // A mask to hold the voxel type for each face on the current slice
            var mask = new byte[ChunkData.Size, ChunkData.Size];

            // Sweep along the current axis
            for (x[(int)axis] = -1; x[(int)axis] < ChunkData.Size;)
            {
                // 1. BUILD THE MASK for the current slice
                for (x[vAxis] = 0; x[vAxis] < ChunkData.Size; ++x[vAxis])
                {
                    for (x[uAxis] = 0; x[uAxis] < ChunkData.Size; ++x[uAxis])
                    {
                        byte type1 = GetVoxel(x[0], x[1], x[2]);
                        byte type2 = GetVoxel(x[0] + dir.X, x[1] + dir.Y, x[2] + dir.Z);

                        // Culling check: face is visible if one side is solid and the other is air.
                        // We also check that they are not the same type, allowing for different materials to have faces.
                        bool solid1 = type1 != 0;
                        bool solid2 = type2 != 0;

                        if (solid1 == solid2)
                        {
                            mask[x[uAxis], x[vAxis]] = 0;
                        }
                        else if (solid1)
                        {
                            mask[x[uAxis], x[vAxis]] = type1;
                        }
                        else
                        {
                            mask[x[uAxis], x[vAxis]] = type2;
                        }
                    }
                }

                ++x[(int)axis];

                // 2. GENERATE MESH from the mask using the greedy algorithm
                for (int j = 0; j < ChunkData.Size; ++j)
                {
                    for (int i = 0; i < ChunkData.Size;)
                    {
                        byte currentType = mask[i, j];
                        if (currentType != 0)
                        {
                            // Find the width of the quad
                            int w;
                            for (w = 1; i + w < ChunkData.Size && mask[i + w, j] == currentType; ++w) { }

                            // Find the height of the quad
                            int h;
                            bool done = false;
                            for (h = 1; j + h < ChunkData.Size; ++h)
                            {
                                for (int k = 0; k < w; ++k)
                                {
                                    if (mask[i + k, j + h] != currentType)
                                    {
                                        done = true;
                                        break;
                                    }
                                }
                                if (done) break;
                            }

                            // Determine if this is a back-face, which needs opposite winding.
                            // This relies on our earlier culling check logic.
                            x[uAxis] = i;
                            x[vAxis] = j;
                            bool backFace = GetVoxel(x[0], x[1], x[2]) == 0;

                            // Set color based on voxel type
                            Color color = currentType switch
                            {
                                1 => new Color("8b4513"), // Dirt
                                2 => new Color("808080"), // Stone
                                _ => Colors.HotPink
                            };
                            st.SetColor(color);

                            // Define the 4 vertices of the quad in 3D space
                            var v1 = new Vector3(); // Bottom-Left
                            var v2 = new Vector3(); // Bottom-Right
                            var v3 = new Vector3(); // Top-Left
                            var v4 = new Vector3(); // Top-Right

                            v1[(int)axis] = v2[(int)axis] = v3[(int)axis] = v4[(int)axis] = x[(int)axis];

                            v1[uAxis] = i; v1[vAxis] = j;
                            v2[uAxis] = i + w; v2[vAxis] = j;
                            v3[uAxis] = i; v3[vAxis] = j + h;
                            v4[uAxis] = i + w; v4[vAxis] = j + h;

                            // Add the two triangles that make up the quad, with correct winding
                            if (backFace)
                            {
                                // Counter-Clockwise winding
                                st.AddVertex(v1); st.AddVertex(v3); st.AddVertex(v4);
                                st.AddVertex(v1); st.AddVertex(v4); st.AddVertex(v2);
                            }
                            else
                            {
                                // Clockwise winding
                                st.AddVertex(v1); st.AddVertex(v4); st.AddVertex(v3);
                                st.AddVertex(v1); st.AddVertex(v2); st.AddVertex(v4);
                            }

                            // Zero out the mask area we just covered
                            for (int l = 0; l < h; ++l)
                                for (int k = 0; k < w; ++k)
                                    mask[i + k, j + l] = 0;

                            i += w;
                        }
                        else
                        {
                            i++;
                        }
                    }
                }
            }
        }

        st.GenerateNormals();
        return st.Commit();
    }

    public byte GetVoxel(int x, int y, int z)
    {
        if (x >= 0 && x < Size && y >= 0 && y < Size && z >= 0 && z < Size)
        {
            return _voxels[(y * Size * Size) + (z * Size) + x];
        }

        var worldPos = new Vector3I(
            Position.X * Size + x,
            Position.Y * Size + y,
            Position.Z * Size + z
        );

        var neighborChunkPos = new Vector3I(
            (int)Mathf.Floor(worldPos.X / (float)Size),
            (int)Mathf.Floor(worldPos.Y / (float)Size),
            (int)Mathf.Floor(worldPos.Z / (float)Size)
        );

        if (_world.VoxelChunks.TryGetValue(neighborChunkPos, out ChunkData neighborChunk))
        {
            var localPos = new Vector3I(
                (worldPos.X % Size + Size) % Size,
                (worldPos.Y % Size + Size) % Size,
                (worldPos.Z % Size + Size) % Size
            );
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
        int stoneLevel = Size / 3;
        int dirtLevel = (Size * 2) / 3;

        for (int x = 0; x < Size; x++)
        {
            for (int z = 0; z < Size; z++)
            {
                int worldX = x + Position.X * Size;
                int worldZ = z + Position.Z * Size;

                float noiseValue = noise.GetNoise2D(worldX, worldZ);
                int surfaceHeight = dirtLevel + (int)(noiseValue * 5);

                for (int y = 0; y < Size; y++)
                {
                    if (y > surfaceHeight)
                    {
                        SetVoxel(x, y, z, Air);
                    }
                    else if (y < stoneLevel)
                    {
                        SetVoxel(x, y, z, Stone);
                    }
                    else
                    {
                        SetVoxel(x, y, z, Dirt);
                    }
                }
            }
        }
    }
}
