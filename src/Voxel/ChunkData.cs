using Godot;
using System.Collections.Generic;
using System.Threading.Tasks;

public partial class ChunkData : GodotObject
{
    public const int Size = 32;
    private const int PaddedSize = Size + 2;
    public Godot.Vector3I Position { get; private set; }

    private readonly World _world;
    private readonly byte[] _voxels = new byte[Size * Size * Size];

    private MeshInstance3D _meshInstance;
    private const byte Air = 0;
    private const byte Dirt = 1;
    private const byte Stone = 2;

    private readonly struct ThreadSafeMeshResult
    {
        public readonly List<System.Numerics.Vector3> Vertices;
        public readonly List<System.Numerics.Vector3> Normals;
        public readonly List<System.Drawing.Color> Colors;

        public ThreadSafeMeshResult(List<System.Numerics.Vector3> v, List<System.Numerics.Vector3> n, List<System.Drawing.Color> c)
        {
            Vertices = v;
            Normals = n;
            Colors = c;
        }
    }

    public ChunkData(World world, Godot.Vector3I position)
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

    private bool _isMeshing = false;

    public async Task RebuildMesh()
    {
        if (_isMeshing) return;
        _isMeshing = true;

        try
        {
            byte[] paddedVoxels = new byte[PaddedSize * PaddedSize * PaddedSize];
            for (int x = 0; x < PaddedSize; x++)
            {
                for (int y = 0; y < PaddedSize; y++)
                {
                    for (int z = 0; z < PaddedSize; z++)
                    {
                        paddedVoxels[(y * PaddedSize * PaddedSize) + (z * PaddedSize) + x] =
                            GetVoxel(x - 1, y - 1, z - 1);
                    }
                }
            }

            ThreadSafeMeshResult meshData = await Task.Run(() => GenerateMeshData(paddedVoxels));
            Callable.From(() => ApplyMeshData(meshData)).CallDeferred();
        }
        catch (System.Exception e)
        {
            GD.PrintErr($"Meshing error in chunk {Position}: {e.Message}");
            _isMeshing = false;
        }
    }

    private void ApplyMeshData(ThreadSafeMeshResult meshData)
    {
        if (_meshInstance != null)
        {
            var newMesh = new ArrayMesh();
            if (meshData.Vertices.Count > 0)
            {
                var godotVertices = new Godot.Collections.Array<Godot.Vector3>();
                var godotNormals = new Godot.Collections.Array<Godot.Vector3>();
                var godotColors = new Godot.Collections.Array<Godot.Color>();

                foreach (var v in meshData.Vertices) godotVertices.Add(new Godot.Vector3(v.X, v.Y, v.Z));
                foreach (var n in meshData.Normals) godotNormals.Add(new Godot.Vector3(n.X, n.Y, n.Z));
                foreach (var c in meshData.Colors) godotColors.Add(new Godot.Color(c.R / 255f, c.G / 255f, c.B / 255f, c.A / 255f));

                var arrays = new Godot.Collections.Array();
                arrays.Resize((int)Mesh.ArrayType.Max);
                arrays[(int)Mesh.ArrayType.Vertex] = godotVertices;
                arrays[(int)Mesh.ArrayType.Normal] = godotNormals;
                arrays[(int)Mesh.ArrayType.Color] = godotColors;
                newMesh.AddSurfaceFromArrays(Mesh.PrimitiveType.Triangles, arrays);
            }
            _meshInstance.Mesh = newMesh;
        }
        _isMeshing = false;
    }

    private static ThreadSafeMeshResult GenerateMeshData(byte[] paddedVoxels)
    {
        var vertices = new List<System.Numerics.Vector3>();
        var normals = new List<System.Numerics.Vector3>();
        var colors = new List<System.Drawing.Color>();

        for (int axis = 0; axis < 3; ++axis)
        {
            int uAxis = (axis + 1) % 3;
            int vAxis = (axis + 2) % 3;
            var dir = new int[3];
            dir[axis] = 1;
            var x = new int[3];
            var mask = new byte[Size, Size];

            for (x[axis] = 0; x[axis] < Size; ++x[axis])
            {
                for (x[vAxis] = 0; x[vAxis] < Size; ++x[vAxis])
                {
                    for (x[uAxis] = 0; x[uAxis] < Size; ++x[uAxis])
                    {
                        int p1_x = x[0] + 1; int p1_y = x[1] + 1; int p1_z = x[2] + 1;
                        int p2_x = p1_x + dir[0]; int p2_y = p1_y + dir[1]; int p2_z = p1_z + dir[2];

                        byte type1 = paddedVoxels[p1_y * PaddedSize * PaddedSize + p1_z * PaddedSize + p1_x];
                        byte type2 = paddedVoxels[p2_y * PaddedSize * PaddedSize + p2_z * PaddedSize + p2_x];

                        bool solid1 = type1 != Air;
                        bool solid2 = type2 != Air;

                        mask[x[uAxis], x[vAxis]] = solid1 != solid2 ? (solid1 ? type1 : type2) : Air;
                    }
                }

                for (int j = 0; j < Size; ++j)
                {
                    for (int i = 0; i < Size;)
                    {
                        byte currentType = mask[i, j];
                        if (currentType != Air)
                        {
                            int w; for (w = 1; i + w < Size && mask[i + w, j] == currentType; ++w) { }
                            int h; bool done = false;
                            for (h = 1; j + h < Size; ++h)
                            {
                                for (int k = 0; k < w; ++k) { if (mask[i + k, j + h] != currentType) { done = true; break; } }
                                if (done) break;
                            }

                            x[uAxis] = i;
                            x[vAxis] = j;

                            int p_x = x[0] + 1; int p_y = x[1] + 1; int p_z = x[2] + 1;
                            bool isSolidBlock = paddedVoxels[p_y * PaddedSize * PaddedSize + p_z * PaddedSize + p_x] != Air;

                            System.Drawing.Color color = currentType switch
                            {
                                Dirt => System.Drawing.Color.FromArgb(255, 139, 69, 19),
                                Stone => System.Drawing.Color.FromArgb(255, 128, 128, 128),
                                _ => System.Drawing.Color.HotPink
                            };

                            var du = new System.Numerics.Vector3();
                            var dv = new System.Numerics.Vector3();
                            switch (uAxis)
                            {
                                case 0: du.X = w; break;
                                case 1: du.Y = w; break;
                                case 2: du.Z = w; break;
                            }
                            switch (vAxis)
                            {
                                case 0: dv.X = h; break;
                                case 1: dv.Y = h; break;
                                case 2: dv.Z = h; break;
                            }

                            var v_bl = new System.Numerics.Vector3(x[0], x[1], x[2]);
                            var v_br = v_bl + du;
                            var v_tl = v_bl + dv;
                            var v_tr = v_bl + du + dv;

                            var normal = new System.Numerics.Vector3(dir[0], dir[1], dir[2]);

                            if (isSolidBlock)
                            {
                                vertices.Add(v_bl); vertices.Add(v_tl); vertices.Add(v_br);
                                vertices.Add(v_br); vertices.Add(v_tl); vertices.Add(v_tr);
                                for (int m = 0; m < 6; m++) normals.Add(normal);
                            }
                            else
                            {
                                vertices.Add(v_bl); vertices.Add(v_br); vertices.Add(v_tl);
                                vertices.Add(v_tl); vertices.Add(v_br); vertices.Add(v_tr);
                                for (int m = 0; m < 6; m++) normals.Add(-normal);
                            }

                            for (int m = 0; m < 6; m++) colors.Add(color);
                            for (int l = 0; l < h; ++l) for (int k = 0; k < w; ++k) mask[i + k, j + l] = 0;
                            i += w;
                        }
                        else i++;
                    }
                }
            }
        }
        return new ThreadSafeMeshResult(vertices, normals, colors);
    }

    public byte GetVoxel(int x, int y, int z)
    {
        if (x >= 0 && x < Size && y >= 0 && y < Size && z >= 0 && z < Size)
        {
            return _voxels[(y * Size * Size) + (z * Size) + x];
        }
        var worldPos = new Godot.Vector3I(Position.X * Size + x, Position.Y * Size + y, Position.Z * Size + z);
        var neighborChunkPos = new Godot.Vector3I(
            (int)Mathf.Floor(worldPos.X / (float)Size),
            (int)Mathf.Floor(worldPos.Y / (float)Size),
            (int)Mathf.Floor(worldPos.Z / (float)Size));

        if (_world != null && _world.VoxelChunks.TryGetValue(neighborChunkPos, out ChunkData neighborChunk))
        {
            var localPos = new Godot.Vector3I(
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

