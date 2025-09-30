using Godot;
using System.Threading;

public static class TerrainGenerator
{
    public static byte[] GenerateVoxelData(Vector3I position, FastNoiseLite terrainNoise, FastNoiseLite caveNoise, CancellationToken token)
    {
        const int size = Chunk.Size;
        byte[] voxels = new byte[size * size * size];

        const float worldScale = 0.1f;
        for (int x = 0; x < size; x++)
        {
            for (int z = 0; z < size; z++)
            {
                if (token.IsCancellationRequested) return null;

                int worldVoxelX = position.X * size + x;
                int worldVoxelZ = position.Z * size + z;
                float scaledX = worldVoxelX * worldScale;
                float scaledZ = worldVoxelZ * worldScale;

                const int seaLevel = 0;
                const int terrainAmplitude = 200;

                float terrainValue = terrainNoise.GetNoise2D(scaledX, scaledZ);
                int surfaceHeight = seaLevel + (int)(terrainValue * terrainAmplitude);

                for (int y = 0; y < size; y++)
                {
                    int worldVoxelY = position.Y * size + y;
                    float scaledY = worldVoxelY * worldScale;

                    float caveValue = caveNoise.GetNoise3D(scaledX, scaledY, scaledZ);
                    if (caveValue > 0.6f)
                    {
                        voxels[(z * size * size) + (y * size) + x] = VoxelTypes.Air;
                        continue;
                    }

                    byte voxelType;
                    if (worldVoxelY > surfaceHeight)
                    {
                        voxelType = (worldVoxelY <= seaLevel) ? VoxelTypes.Water : VoxelTypes.Air;
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
                    voxels[(z * size * size) + (y * size) + x] = voxelType;
                }
            }
        }
        return voxels;
    }
}
