#if TOOLS
using Godot;
using System.Collections.Generic;

[Tool]
public partial class ChunkNode3DGizmoPlugin : EditorNode3DGizmoPlugin
{
    public ChunkNode3DGizmoPlugin()
    {
        CreateMaterial("lines_material", Colors.Yellow);
        
    }

    public override string _GetGizmoName()
    {
        return "Chunk Boundaries";
    }

    public override bool _HasGizmo(Node3D forNode)
    {
        // This gizmo will only appear for nodes of type World
        return forNode is World;
    }

    public override void _Redraw(EditorNode3DGizmo gizmo)
    {
        gizmo.Clear();
        var world = gizmo.GetNode3D() as World;
        if (world == null) return;
        
        var material = GetMaterial("lines_material", gizmo);
        var lines = new List<Vector3>();
        float chunkSize = ChunkData.Size;

        foreach (var chunkPosition in world.VoxelChunks.Keys)
        {
            var pos = chunkPosition * (int)chunkSize;
            // Create the 12 edges of a cube
            lines.Add(new Vector3(pos.X, pos.Y, pos.Z)); lines.Add(new Vector3(pos.X + chunkSize, pos.Y, pos.Z));
            lines.Add(new Vector3(pos.X + chunkSize, pos.Y, pos.Z)); lines.Add(new Vector3(pos.X + chunkSize, pos.Y, pos.Z + chunkSize));
            lines.Add(new Vector3(pos.X + chunkSize, pos.Y, pos.Z + chunkSize)); lines.Add(new Vector3(pos.X, pos.Y, pos.Z + chunkSize));
            lines.Add(new Vector3(pos.X, pos.Y, pos.Z + chunkSize)); lines.Add(new Vector3(pos.X, pos.Y, pos.Z));
            
            lines.Add(new Vector3(pos.X, pos.Y + chunkSize, pos.Z)); lines.Add(new Vector3(pos.X + chunkSize, pos.Y + chunkSize, pos.Z));
            lines.Add(new Vector3(pos.X + chunkSize, pos.Y + chunkSize, pos.Z)); lines.Add(new Vector3(pos.X + chunkSize, pos.Y + chunkSize, pos.Z + chunkSize));
            lines.Add(new Vector3(pos.X + chunkSize, pos.Y + chunkSize, pos.Z + chunkSize)); lines.Add(new Vector3(pos.X, pos.Y + chunkSize, pos.Z + chunkSize));
            lines.Add(new Vector3(pos.X, pos.Y + chunkSize, pos.Z + chunkSize)); lines.Add(new Vector3(pos.X, pos.Y + chunkSize, pos.Z));

            lines.Add(new Vector3(pos.X, pos.Y, pos.Z)); lines.Add(new Vector3(pos.X, pos.Y + chunkSize, pos.Z));
            lines.Add(new Vector3(pos.X + chunkSize, pos.Y, pos.Z)); lines.Add(new Vector3(pos.X + chunkSize, pos.Y + chunkSize, pos.Z));
            lines.Add(new Vector3(pos.X + chunkSize, pos.Y, pos.Z + chunkSize)); lines.Add(new Vector3(pos.X + chunkSize, pos.Y + chunkSize, pos.Z + chunkSize));
            lines.Add(new Vector3(pos.X, pos.Y, pos.Z + chunkSize)); lines.Add(new Vector3(pos.X, pos.Y + chunkSize, pos.Z + chunkSize));
        }

        gizmo.AddLines(lines.ToArray(), material);
    }
}
#endif
