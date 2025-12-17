using Godot;
using System;
using System.Buffers;
using System.Threading;

public class GreedyMesher : BaseMesher
{
    public MeshData GenerateMeshData(MeshJobData jobData, VoxelDefinition[] definitions, CancellationToken token)
    {
        const int size = Chunk.Size;
        const int paddedSize = size + 2;
        int maxVoxels = paddedSize * paddedSize * paddedSize;

        // Rent buffers for the algorithm
        byte[] paddedVoxels = ArrayPool<byte>.Shared.Rent(maxVoxels);
        
        // Initial capacity estimates (can grow)
        // Worst case is theoretical, but usually much lower. Start with 4096.
        int initialCap = 4096;
        Vector3[] vertBuffer = ArrayPool<Vector3>.Shared.Rent(initialCap);
        Vector3[] normBuffer = ArrayPool<Vector3>.Shared.Rent(initialCap);
        Color[] colBuffer = ArrayPool<Color>.Shared.Rent(initialCap);
        int[] idxBuffer = ArrayPool<int>.Shared.Rent(initialCap * 3 / 2); // Indices usually 1.5x vertices

        int vCount = 0;
        int iCount = 0;

        try
        {
            BuildPaddedVoxelsFromJobData(jobData, paddedVoxels);

            // Reuse mask buffer
            var mask = ArrayPool<(byte type, bool forwardFace)>.Shared.Rent(size * size);

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

                                // --- Add Quad ---
                                EnsureCapacity(ref vertBuffer, vCount + 4);
                                EnsureCapacity(ref normBuffer, vCount + 4);
                                EnsureCapacity(ref colBuffer, vCount + 4);
                                EnsureCapacity(ref idxBuffer, iCount + 6);

                                vertBuffer[vCount] = basePos;
                                vertBuffer[vCount+1] = basePos + du;
                                vertBuffer[vCount+2] = basePos + dv;
                                vertBuffer[vCount+3] = basePos + du + dv;

                                Vector3 faceNormal = forwardFace ? normal : -normal;
                                normBuffer[vCount] = faceNormal;
                                normBuffer[vCount+1] = faceNormal;
                                normBuffer[vCount+2] = faceNormal;
                                normBuffer[vCount+3] = faceNormal;

                                Color col = definitions[type].VoxelColor;
                                colBuffer[vCount] = col;
                                colBuffer[vCount+1] = col;
                                colBuffer[vCount+2] = col;
                                colBuffer[vCount+3] = col;

                                if (forwardFace) // CCW
                                {
                                    idxBuffer[iCount++] = vCount + 0;
                                    idxBuffer[iCount++] = vCount + 2;
                                    idxBuffer[iCount++] = vCount + 1;
                                    idxBuffer[iCount++] = vCount + 1;
                                    idxBuffer[iCount++] = vCount + 2;
                                    idxBuffer[iCount++] = vCount + 3;
                                }
                                else // CW
                                {
                                    idxBuffer[iCount++] = vCount + 0;
                                    idxBuffer[iCount++] = vCount + 1;
                                    idxBuffer[iCount++] = vCount + 2;
                                    idxBuffer[iCount++] = vCount + 1;
                                    idxBuffer[iCount++] = vCount + 3;
                                    idxBuffer[iCount++] = vCount + 2;
                                }
                                vCount += 4;

                                // Clear Mask
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
            } // End Axis Loop

            ArrayPool<(byte type, bool forwardFace)>.Shared.Return(mask);

            // Copy to final arrays
            var finalVerts = new Vector3[vCount];
            var finalNorms = new Vector3[vCount];
            var finalColors = new Color[vCount];
            var finalIndices = new int[iCount];

            Array.Copy(vertBuffer, finalVerts, vCount);
            Array.Copy(normBuffer, finalNorms, vCount);
            Array.Copy(colBuffer, finalColors, vCount);
            Array.Copy(idxBuffer, finalIndices, iCount);

            return new MeshData 
            { 
                Vertices = finalVerts, 
                Normals = finalNorms, 
                Colors = finalColors, 
                Indices = finalIndices 
            };
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(paddedVoxels);
            ArrayPool<Vector3>.Shared.Return(vertBuffer);
            ArrayPool<Vector3>.Shared.Return(normBuffer);
            ArrayPool<Color>.Shared.Return(colBuffer);
            ArrayPool<int>.Shared.Return(idxBuffer);
        }
    }

    private void EnsureCapacity<T>(ref T[] array, int needed)
    {
        if (needed > array.Length)
        {
            int newSize = array.Length * 2;
            while (newSize < needed) newSize *= 2;
            
            var newArray = ArrayPool<T>.Shared.Rent(newSize);
            Array.Copy(array, newArray, array.Length);
            ArrayPool<T>.Shared.Return(array);
            array = newArray;
        }
    }

    private void BuildPaddedVoxelsFromJobData(MeshJobData data, byte[] paddedVoxels)
    {
        const int size = Chunk.Size;
        const int paddedSize = size + 2;

        byte[] centerVoxels = data.CenterVoxels;
        byte[] rightVoxels = data.NeighborVoxels[0];
        byte[] leftVoxels = data.NeighborVoxels[1];
        byte[] upVoxels = data.NeighborVoxels[2];
        byte[] downVoxels = data.NeighborVoxels[3];
        byte[] frontVoxels = data.NeighborVoxels[4];
        byte[] backVoxels = data.NeighborVoxels[5];

        // Optimized fill? For now, standard loop is fine as it's just array access.
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
                        voxelValue = centerVoxels[(z * size * size) + (y * size) + x];
                    }
                    else
                    {
                        int localX = (x + size) % size;
                        int localY = (y + size) % size;
                        int localZ = (z + size) % size;
                        int neighborIndex = (localZ * size * size) + (localY * size) + localX;

                        if (x < 0 && leftVoxels != null) voxelValue = leftVoxels[neighborIndex];
                        else if (x >= size && rightVoxels != null) voxelValue = rightVoxels[neighborIndex];
                        else if (y < 0 && downVoxels != null) voxelValue = downVoxels[neighborIndex];
                        else if (y >= size && upVoxels != null) voxelValue = upVoxels[neighborIndex];
                        else if (z < 0 && backVoxels != null) voxelValue = backVoxels[neighborIndex];
                        else if (z >= size && frontVoxels != null) voxelValue = frontVoxels[neighborIndex];
                        else voxelValue = VoxelTypes.Air;
                    }
                    paddedVoxels[paddedIndex] = voxelValue;
                }
            }
        }
    }
}