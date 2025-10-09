using Godot;
using System.Collections.Generic;
using System.Threading;

public class GreedyMesher : BaseMesher
{

    public MeshData GenerateMeshData(MeshJobData jobData, VoxelDefinition[] definitions, CancellationToken token)
    {
        const int size = Chunk.Size;
        const int paddedSize = size + 2;

        byte[] paddedVoxels = BuildPaddedVoxelsFromJobData(jobData);

        var vertices = new List<Vector3>();
        var normals = new List<Vector3>();
        var colors = new List<Color>();
        var indices = new List<int>();
        int indexCount = 0;

        int Idx(int x, int y, int z) => (z * paddedSize * paddedSize) + (y * paddedSize) + x;

        // --- Sweep over 3 axes (X, Y, Z) ---
        for (int axis = 0; axis < 3; axis++)
        {
            int u = (axis + 1) % 3;
            int v = (axis + 2) % 3;
            var x = new int[3];
            var q = new int[3];
            q[axis] = 1;

            var normal = new Vector3();
            normal[axis] = 1;

            var mask = new (byte type, bool forwardFace)[size * size];

            for (x[axis] = -1; x[axis] < size; x[axis]++)
            {
                token.ThrowIfCancellationRequested();

                // 1. --- Build the 2D mask for the current slice ---
                int n = 0;
                for (x[v] = 0; x[v] < size; x[v]++)
                {
                    for (x[u] = 0; x[u] < size; x[u]++)
                    {
                        byte typeA = paddedVoxels[Idx(x[0] + 1, x[1] + 1, x[2] + 1)];
                        byte typeB = paddedVoxels[Idx(x[0] + q[0] + 1, x[1] + q[1] + 1, x[2] + q[2] + 1)];

                        bool solidA = definitions[typeA].IsSolid;
                        bool solidB = definitions[typeB].IsSolid;

                        if (solidA == solidB)
                        {
                            mask[n++] = (0, false);
                        }
                        else
                        {
                            mask[n++] = solidA ? (typeA, true) : (typeB, false);
                        }
                    }
                }

                // 2. --- Generate greedy quads from the mask ---
                n = 0;
                for (int j = 0; j < size; j++)
                {
                    for (int i = 0; i < size;)
                    {
                        var (type, forwardFace) = mask[n];
                        if (type != 0)
                        {
                            int w;
                            for (w = 1; i + w < size && mask[n + w].type == type && mask[n + w].forwardFace == forwardFace; w++) { }

                            int h;
                            bool done = false;
                            for (h = 1; j + h < size; h++)
                            {
                                for (int k = 0; k < w; k++)
                                {
                                    if (mask[n + k + h * size].type != type || mask[n + k + h * size].forwardFace != forwardFace)
                                    {
                                        done = true; break;
                                    }
                                }
                                if (done) break;
                            }

                            x[u] = i;
                            x[v] = j;

                            var du = new Vector3(); du[u] = w;
                            var dv = new Vector3(); dv[v] = h;

                            var basePos = new Vector3(x[0], x[1], x[2]);
                            basePos[axis] += 1;

                            vertices.Add(basePos);          // v0 (bottom-left)
                            vertices.Add(basePos + du);     // v1 (bottom-right)
                            vertices.Add(basePos + dv);     // v2 (top-left)
                            vertices.Add(basePos + du + dv);// v3 (top-right)

                            Vector3 faceNormal = forwardFace ? normal : -normal;
                            normals.Add(faceNormal); normals.Add(faceNormal); normals.Add(faceNormal); normals.Add(faceNormal);

                            Color col = definitions[type].VoxelColor;
                            colors.Add(col); colors.Add(col); colors.Add(col); colors.Add(col);

                            if (forwardFace) // Counter-Clockwise winding
                            {
                                indices.Add(indexCount + 0);
                                indices.Add(indexCount + 2);
                                indices.Add(indexCount + 1);
                                indices.Add(indexCount + 1);
                                indices.Add(indexCount + 2);
                                indices.Add(indexCount + 3);
                            }
                            else // Clockwise winding
                            {
                                indices.Add(indexCount + 0);
                                indices.Add(indexCount + 1);
                                indices.Add(indexCount + 2);
                                indices.Add(indexCount + 1);
                                indices.Add(indexCount + 3);
                                indices.Add(indexCount + 2);
                            }
                            indexCount += 4;

                            for (int l = 0; l < h; l++)
                                for (int k = 0; k < w; k++)
                                    mask[n + k + l * size] = (0, false);

                            i += w;
                            n += w;
                        }
                        else
                        {
                            i++;
                            n++;
                        }
                    }
                }
            }
        }
        return new MeshData { Vertices = vertices, Normals = normals, Colors = colors, Indices = indices };
    }

    private byte[] BuildPaddedVoxelsFromJobData(MeshJobData data)
    {
        const int size = Chunk.Size;
        const int paddedSize = size + 2;
        var paddedVoxels = new byte[paddedSize * paddedSize * paddedSize];

        byte[] centerVoxels = data.CenterVoxels;
        byte[] rightVoxels = data.NeighborVoxels[0];
        byte[] leftVoxels = data.NeighborVoxels[1];
        byte[] upVoxels = data.NeighborVoxels[2];
        byte[] downVoxels = data.NeighborVoxels[3];
        byte[] frontVoxels = data.NeighborVoxels[4];
        byte[] backVoxels = data.NeighborVoxels[5];

        for (int x = -1; x < size + 1; x++)
        {
            for (int y = -1; y < size + 1; y++)
            {
                for (int z = -1; z < size + 1; z++)
                {
                    int paddedIndex = ((z + 1) * paddedSize * paddedSize) + ((y + 1) * paddedSize) + (x + 1);
                    byte voxelValue;

                    if (x >= 0 && x < size && y >= 0 && y < size && z >= 0 && z < size)
                    {
                        // Case 1: Voxel is inside the main chunk's data.
                        voxelValue = centerVoxels[(z * size * size) + (y * size) + x];
                    }
                    else
                    {
                        // Case 2: Voxel is in the padding (a neighbor chunk).
                        int localX = (x + size) % size;
                        int localY = (y + size) % size;
                        int localZ = (z + size) % size;
                        int neighborIndex = (localZ * size * size) + (localY * size) + localX;

                        // Check which neighbor this coordinate falls into.
                        if (x < 0 && leftVoxels != null) voxelValue = leftVoxels[neighborIndex];
                        else if (x >= size && rightVoxels != null) voxelValue = rightVoxels[neighborIndex];
                        else if (y < 0 && downVoxels != null) voxelValue = downVoxels[neighborIndex];
                        else if (y >= size && upVoxels != null) voxelValue = upVoxels[neighborIndex];
                        else if (z < 0 && backVoxels != null) voxelValue = backVoxels[neighborIndex];
                        else if (z >= size && frontVoxels != null) voxelValue = frontVoxels[neighborIndex];
                        else voxelValue = VoxelTypes.Air; // Fallback if a neighbor doesn't exist.
                    }
                    paddedVoxels[paddedIndex] = voxelValue;
                }
            }
        }
        return paddedVoxels;
    }
}
