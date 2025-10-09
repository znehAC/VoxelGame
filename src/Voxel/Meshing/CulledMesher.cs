using Godot;
using System.Collections.Generic;
using System.Threading;

public class CulledMesher : BaseMesher
{
    public MeshData GenerateMeshData(MeshJobData jobData, VoxelDefinition[] definitions, CancellationToken token)
    {
        var vertices = new List<Vector3>();
        var normals = new List<Vector3>();
        var colors = new List<Color>();
        var indices = new List<int>();
        int indexCount = 0;

        byte[] voxels = jobData.CenterVoxels;
        const int size = Chunk.Size;

        for (int x = 0; x < size; x++)
        {
            for (int y = 0; y < size; y++)
            {
                for (int z = 0; z < size; z++)
                {
                    token.ThrowIfCancellationRequested();

                    var voxelType = voxels[(z * size * size) + (y * size) + x];
                    if (!definitions[voxelType].IsSolid) continue;

                    var blockPos = new Vector3(x, y, z);

                    for (int axis = 0; axis < 3; axis++)
                    {
                        for (int direction = -1; direction <= 1; direction += 2)
                        {
                            var q = new Vector3I();
                            q[axis] = direction;

                            var neighborPos = new Vector3I(x, y, z) + q;
                            byte neighborType = GetVoxelFromJobData(jobData, neighborPos.X, neighborPos.Y, neighborPos.Z);

                            if (definitions[neighborType].IsSolid) continue;

                            int u = (axis + 1) % 3;
                            int v = (axis + 2) % 3;

                            var du = new Vector3();
                            du[u] = 1;

                            var dv = new Vector3();
                            dv[v] = 1;

                            var facePos = blockPos;
                            if (direction == 1) facePos[axis] += 1;

                            vertices.Add(facePos);          // v0 (origin)
                            vertices.Add(facePos + du);     // v1 (along u)
                            vertices.Add(facePos + dv);     // v2 (along v)
                            vertices.Add(facePos + du + dv);// v3 (diagonal)

                            var normal = new Vector3();
                            normal[axis] = direction;
                            normals.Add(normal); normals.Add(normal);
                            normals.Add(normal); normals.Add(normal);

                            var color = definitions[voxelType].VoxelColor;
                            colors.Add(color); colors.Add(color);
                            colors.Add(color); colors.Add(color);

                            if (direction == 1) // Positive direction faces
                            {
                                indices.Add(indexCount + 0);
                                indices.Add(indexCount + 2);
                                indices.Add(indexCount + 1);
                                indices.Add(indexCount + 1);
                                indices.Add(indexCount + 2);
                                indices.Add(indexCount + 3);
                            }
                            else // Negative direction faces
                            {
                                indices.Add(indexCount + 0);
                                indices.Add(indexCount + 1);
                                indices.Add(indexCount + 2);
                                indices.Add(indexCount + 1);
                                indices.Add(indexCount + 3);
                                indices.Add(indexCount + 2);
                            }
                            indexCount += 4;
                        }
                    }
                }
            }
        }
        return new MeshData { Vertices = vertices, Normals = normals, Colors = colors, Indices = indices };
    }

    private byte GetVoxelFromJobData(MeshJobData data, int x, int y, int z)
    {
        const int size = Chunk.Size;
        if (x >= 0 && x < size && y >= 0 && y < size && z >= 0 && z < size)
        {
            return data.CenterVoxels[(z * size * size) + (y * size) + x];
        }

        int localX = (x + size) % size;
        int localY = (y + size) % size;
        int localZ = (z + size) % size;
        int neighborIndex = (localZ * size * size) + (localY * size) + localX;

        if (x < 0) return data.NeighborVoxels[1]?[neighborIndex] ?? 0;
        if (x >= size) return data.NeighborVoxels[0]?[neighborIndex] ?? 0;
        if (y < 0) return data.NeighborVoxels[3]?[neighborIndex] ?? 0;
        if (y >= size) return data.NeighborVoxels[2]?[neighborIndex] ?? 0;
        if (z < 0) return data.NeighborVoxels[5]?[neighborIndex] ?? 0;
        if (z >= size) return data.NeighborVoxels[4]?[neighborIndex] ?? 0;

        return 0;
    }
}
