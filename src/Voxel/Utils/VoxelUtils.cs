using Godot;

public static class VoxelUtils
{
    public static readonly Vector3I[] NeighborOffsets =
    {
        new(1, 0, 0),  // Right
        new(-1, 0, 0), // Left
        new(0, 1, 0),  // Up
        new(0, -1, 0), // Down
        new(0, 0, 1),  // Forward
        new(0, 0, -1)  // Back
    };
}
