using Godot;
using System.Collections.Generic;

public struct MeshData
{
    public List<Vector3> Vertices;
    public List<Vector3> Normals;
    public List<Color> Colors;
    public List<int> Indices;
}

public static class GreedyMesher
{

    public static MeshData GenerateMeshData(byte[] paddedVoxels, int size, Color[] palette)
    {
        int paddedSize = size + 2;
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
                // 1. --- Build the 2D mask for the current slice ---
                int n = 0;
                for (x[v] = 0; x[v] < size; x[v]++)
                {
                    for (x[u] = 0; x[u] < size; x[u]++)
                    {
                        byte typeA = paddedVoxels[Idx(x[0] + 1, x[1] + 1, x[2] + 1)];
                        byte typeB = paddedVoxels[Idx(x[0] + q[0] + 1, x[1] + q[1] + 1, x[2] + q[2] + 1)];

                        bool solidA = typeA != 0;
                        bool solidB = typeB != 0;

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
                            // Find width and height of the quad
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

                            // Add the quad to the mesh data
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

                            Color col = palette[type];
                            colors.Add(col); colors.Add(col); colors.Add(col); colors.Add(col);

                            if (forwardFace) // Counter-Clockwise winding for Godot
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

                            // Zero out the mask area we just covered
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
}
