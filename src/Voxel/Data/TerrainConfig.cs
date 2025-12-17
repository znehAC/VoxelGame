using Godot;

namespace VoxelPowderSim.Src.Voxel.Data;

public partial class TerrainConfig : Resource
{
    public enum WorldMode
    {
        Infinite,
        Fixed
    }

    [Export] public WorldMode Mode { get; set; } = WorldMode.Infinite;
    [Export] public int ChunkSize { get; set; } = 32;
    [Export] public int RenderDistance { get; set; } = 8;
    [Export] public float VoxelScale { get; set; } = 0.1f;
    [Export] public Vector3I FixedBoundsSize { get; set; } = new Vector3I(512, 128, 512);
    [Export] public float OriginShiftThreshold { get; set; } = 10000.0f;
}
