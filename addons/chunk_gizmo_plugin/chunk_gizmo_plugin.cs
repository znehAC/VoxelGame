#if TOOLS
using Godot;
using System;

[Tool]
public partial class chunk_gizmo_plugin : EditorPlugin
{
    private ChunkNode3DGizmoPlugin _gizmoPlugin; 

	public override void _EnterTree()
	{
        _gizmoPlugin = new ChunkNode3DGizmoPlugin();
        AddNode3DGizmoPlugin(_gizmoPlugin);
	}

	public override void _ExitTree()
	{
        RemoveNode3DGizmoPlugin(_gizmoPlugin);
	}
}
#endif
