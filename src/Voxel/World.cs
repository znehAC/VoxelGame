using Godot;
using System.Collections.Generic;

public partial class World : Node3D
{
    [Export]
    public StandardMaterial3D VoxelMaterial { get; set; }
    [Export]
    public int WorldSeed { get; set; }

    public Dictionary<Vector3I, Chunk> VoxelChunks = new();

    [Export]
    private Label PerformanceLabel { get; set; }

    private Node3D _player;

    private PriorityQueue<Vector3I, float> _dirtyChunkQueue = new();
    private HashSet<Vector3I> _dirtyChunkSet = new();

    private VoxelMesherPool _mesherPool;

    private const int MaxJobsToDispatchPerFrame = 8;
    private const int MaxResultsToProcessPerFrame = 8;

    private FastNoiseLite _terrainNoise = new FastNoiseLite();
    private FastNoiseLite _caveNoise = new FastNoiseLite();

    private bool _debugWireframe = false;

    public override void _EnterTree()
    {
        _player = GetNode<Node3D>("Player");

        _terrainNoise.Seed = (int)GD.Randi();
        _terrainNoise.Frequency = 0.005f;

        _caveNoise.Seed = (int)GD.Randi();
        _caveNoise.Frequency = 0.02f;
        _caveNoise.FractalType = FastNoiseLite.FractalTypeEnum.Ridged;

        _mesherPool = new VoxelMesherPool(this, _terrainNoise, _caveNoise);
        _mesherPool.Start();

        for (int x = 0; x < 5; x++)
        {
            for (int y = -70; y < 20; y++)
            {
                for (int z = 0; z < 5; z++)
                {
                    var position = new Vector3I(x, y, z);
                    var newChunkData = new Chunk(this, position);

                    newChunkData.InitializeMesh(this, VoxelMaterial);
                    VoxelChunks.Add(position, newChunkData);
                    AddDirtyChunk(position);
                }
            }
        }
    }

    public override void _Process(double delta)
    {
        DispatchMeshJobs();
        ProcessMeshResults();
        UpdateMetrics();
    }

    public override void _Input(InputEvent e)
    {
        if (e.IsActionPressed("toggle_debug_wireframe"))
        {
            ToggleWireframe();
        }
    }

    private void DispatchMeshJobs()
    {
        int dispatchedCount = 0;
        int maxQueuedJobs = _mesherPool.WorkerCount * 2;

        while (dispatchedCount < MaxJobsToDispatchPerFrame &&
               _dirtyChunkQueue.Count > 0 &&
               _mesherPool.WorkQueueCount < maxQueuedJobs)
        {
            Vector3I chunkPosition = _dirtyChunkQueue.Dequeue();
            _mesherPool.EnqueueJob(chunkPosition);
            dispatchedCount++;
        }
    }

    private void ProcessMeshResults()
    {
        int processedCount = 0;
        while (processedCount < MaxResultsToProcessPerFrame && _mesherPool.TryDequeueResult(out var result))
        {
            var (position, meshData) = result;
            if (VoxelChunks.TryGetValue(position, out Chunk chunk))
            {
                chunk.ApplyMeshData(meshData);
            }
            _dirtyChunkSet.Remove(position);
            processedCount++;
        }
    }

    public void AddDirtyChunk(Vector3I chunkPosition)
    {
        if (_dirtyChunkSet.Add(chunkPosition))
        {
            float distance = ((Vector3)chunkPosition).DistanceTo(_player.GlobalPosition);
            _dirtyChunkQueue.Enqueue(chunkPosition, distance);
        }
    }

    private void UpdateMetrics()
    {
        if (PerformanceLabel != null)
        {

            double fps = Performance.GetMonitor(Performance.Monitor.TimeFps);
            long vertices = (long)Performance.GetMonitor(Performance.Monitor.RenderTotalPrimitivesInFrame);
            long drawCalls = (long)Performance.GetMonitor(Performance.Monitor.RenderTotalDrawCallsInFrame);
            long dirtyChunks = _dirtyChunkSet.Count;

            PerformanceLabel.Text = $@"
                FPS: {fps:F0}
                Vertices: {vertices}
                Draw Calls: {drawCalls}
                dirty Chunks: {dirtyChunks}";
        }
    }

    private void ToggleWireframe()
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
