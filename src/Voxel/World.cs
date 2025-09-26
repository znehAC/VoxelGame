using Godot;
using System.Collections.Generic;

public partial class World : Node3D
{
    [Export]
    public StandardMaterial3D VoxelMaterial { get; set; }

    public Dictionary<Vector3I, ChunkData> VoxelChunks = new();

    [Export]
    private Label PerformanceLabel { get; set; }

    private FastNoiseLite _noise = new();
    private bool _debugWireframe = false;

    public override void _EnterTree()
    {
        _noise.Seed = (int)GD.Randi();
        _noise.Frequency = 0.005f;
        _noise.FractalType = FastNoiseLite.FractalTypeEnum.Fbm;
        _noise.FractalOctaves = 5;
        _noise.FractalLacunarity = 2.0f;
        _noise.FractalGain = 0.5f;

        for (int x = 0; x < 5; x++)
        {
            for (int z = 0; z < 5; z++)
            {
                var position = new Vector3I(x, 0, z);
                var newChunkData = new ChunkData(this, position);
                newChunkData.GenerateTerrain(_noise);
                VoxelChunks.Add(position, newChunkData);
            }
        }

        foreach (var chunk in VoxelChunks.Values)
        {
            chunk.InitializeMesh(this, VoxelMaterial);
            // chunk.RebuildMesh();
        }
    }

    public override void _Process(double delta)
    {

        foreach (var chunk in VoxelChunks.Values)
        {
            chunk.RebuildMesh();
        }

        if (PerformanceLabel != null)
        {

            double fps = Performance.GetMonitor(Performance.Monitor.TimeFps);
            long vertices = (long)Performance.GetMonitor(Performance.Monitor.RenderTotalPrimitivesInFrame);
            long drawCalls = (long)Performance.GetMonitor(Performance.Monitor.RenderTotalDrawCallsInFrame);

            PerformanceLabel.Text = $"FPS: {fps:F0}\nVertices: {vertices}\nDraw Calls: {drawCalls}";
        }
    }

    public override void _Input(InputEvent e)
    {
        if (e.IsActionPressed("toggle_debug_wireframe"))
        {
            _debugWireframe = !_debugWireframe;
            GD.Print("Wireframe: " + _debugWireframe);

            Viewport viewport = GetViewport();
            viewport.DebugDraw = _debugWireframe
                ? Viewport.DebugDrawEnum.Wireframe
                : Viewport.DebugDrawEnum.Disabled;


            foreach (var child in GetChildren())
            {
                if (child is MeshInstance3D meshInstance)
                {
                    var material = meshInstance.MaterialOverride as StandardMaterial3D;
                    if (material != null)
                    {
                        material.CullMode = _debugWireframe
                            ? BaseMaterial3D.CullModeEnum.Disabled
                            : BaseMaterial3D.CullModeEnum.Back;
                    }
                }
            }
        }
    }

}
