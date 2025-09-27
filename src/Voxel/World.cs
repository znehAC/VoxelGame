using Godot;
using System.Collections.Generic;
using System.Linq;

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
    private VoxelMesherPool _mesherPool;

    private HashSet<Vector3I> _chunksToGenerate = new();
    private PriorityQueue<Vector3I, float> _chunksToMesh = new();
    private HashSet<Vector3I> _dirtyChunkSet = new();

    private const int MaxJobsToDispatchPerFrame = 8;
    private const int MaxResultsToProcessPerFrame = 8;

    private FastNoiseLite _terrainNoise = new FastNoiseLite();
    private FastNoiseLite _caveNoise = new FastNoiseLite();
    private bool _debugWireframe = false;

    public override void _EnterTree()
    {
        _player = GetNode<Node3D>("Player");

        _terrainNoise.Seed = WorldSeed == 0 ? WorldSeed : (int)GD.Randi();
        _terrainNoise.Frequency = 0.005f;

        _caveNoise.Seed = WorldSeed == 0 ? WorldSeed : (int)GD.Randi();
        _caveNoise.Frequency = 0.02f;
        _caveNoise.FractalType = FastNoiseLite.FractalTypeEnum.Ridged;

        _mesherPool = new VoxelMesherPool(this, _terrainNoise, _caveNoise);
        _mesherPool.Start();


        for (int x = 0; x < 15; x++)
        {
            for (int y = 0; y < 30; y++)
            {
                for (int z = 0; z < 15; z++)
                {
                    var position = new Vector3I(x, y, z);
                    var newChunk = new Chunk(this, position);
                    newChunk.InitializeMesh(this, VoxelMaterial);
                    VoxelChunks.Add(position, newChunk);
                    QueueChunkForGeneration(position);
                }
            }
        }
    }

    public override void _Process(double delta)
    {
        DispatchGenerationJobs();
        ProcessResults();
        DispatchMeshingJobs();

        UpdateMetrics();
    }

    public override void _Input(InputEvent e)
    {
        if (e.IsActionPressed("toggle_debug_wireframe"))
        {
            ToggleWireframe();
        }
    }

    public void QueueChunkForGeneration(Vector3I chunkPosition)
    {
        if (_dirtyChunkSet.Add(chunkPosition))
        {
            _chunksToGenerate.Add(chunkPosition);
        }
    }

    private void DispatchGenerationJobs()
    {

        foreach (var position in _chunksToGenerate)
        {
            _mesherPool.EnqueueJob(new VoxelJob(position, VoxelJobType.GenerateData));
        }
        _chunksToGenerate.Clear();
    }

    private void ProcessResults()
    {
        int processedCount = 0;
        while (processedCount < MaxResultsToProcessPerFrame && _mesherPool.TryDequeueResult(out var result))
        {
            var (job, data) = result;

            switch (job.JobType)
            {
                case VoxelJobType.GenerateData:

                    CheckIfReadyToMesh(job.Position);
                    foreach (var offset in VoxelUtils.NeighborOffsets)
                    {
                        CheckIfReadyToMesh(job.Position + offset);
                    }
                    break;

                case VoxelJobType.GenerateMesh:
                    if (VoxelChunks.TryGetValue(job.Position, out Chunk chunk) && data is MeshData meshData)
                    {
                        chunk.ApplyMeshData(meshData);
                        _dirtyChunkSet.Remove(job.Position);
                    }
                    break;
            }
            processedCount++;
        }
    }

    private void CheckIfReadyToMesh(Vector3I position)
    {
        if (!VoxelChunks.TryGetValue(position, out var chunk)
            || !chunk.IsDataGenerated
            || _chunksToMesh.UnorderedItems.Any(item => item.Element == position))
        {
            return;
        }

        foreach (var offset in VoxelUtils.NeighborOffsets)
        {
            var neighborPos = position + offset;

            if (VoxelChunks.TryGetValue(neighborPos, out var neighbor))
            {
                if (!neighbor.IsDataGenerated)
                {
                    return;
                }
            }
        }

        float distance = ((Vector3)position).DistanceTo(_player.GlobalPosition);
        _chunksToMesh.Enqueue(position, distance);
    }

    private void DispatchMeshingJobs()
    {
        int dispatchedCount = 0;
        int maxQueuedJobs = _mesherPool.WorkerCount * 2;

        while (dispatchedCount < MaxJobsToDispatchPerFrame &&
               _chunksToMesh.Count > 0 &&
               _mesherPool.WorkQueueCount < maxQueuedJobs)
        {
            Vector3I chunkPosition = _chunksToMesh.Dequeue();
            _mesherPool.EnqueueJob(new VoxelJob(chunkPosition, VoxelJobType.GenerateMesh));
            dispatchedCount++;
        }
    }

    private void UpdateMetrics()
    {
        if (PerformanceLabel != null)
        {
            double fps = Performance.GetMonitor(Performance.Monitor.TimeFps);
            long vertices = (long)Performance.GetMonitor(Performance.Monitor.RenderTotalPrimitivesInFrame);
            long drawCalls = (long)Performance.GetMonitor(Performance.Monitor.RenderTotalDrawCallsInFrame);

            long totalDirty = _dirtyChunkSet.Count;
            long generating = _chunksToGenerate.Count;
            long meshing = _chunksToMesh.Count;
            long inFlight = _mesherPool.WorkQueueCount;

            PerformanceLabel.Text = $@"
            FPS: {fps:F0}
            Vertices: {vertices}
            Draw Calls: {drawCalls}
            ---
            Dirty (Total): {totalDirty}
            Queued for Gen: {generating}
            Queued for Mesh: {meshing}
            In Worker Queue: {inFlight}";
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
