using Godot;
using System.Collections.Generic;

public static class VoxelUtils
{
    public static readonly Vector3I[] NeighborOffsets =
    {
        Vector3I.Right, Vector3I.Left, Vector3I.Up, Vector3I.Down, Vector3I.Forward, Vector3I.Back
    };

}
