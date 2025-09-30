using Godot;
using System.Collections.Generic;
using System.Linq;
using System.Collections.Concurrent;
using System.Threading;
using System.Diagnostics;

public partial class World : Node3D
{
    [Export] public StandardMaterial3D VoxelMaterial { get; set; }
    [Export] public int LoadRadius { get; set; } = 8;
    [Export] public int WorldHeightInChunks { get; set; } = 10;
    [Export] private Label PerformanceLabel { get; set; }
    [Export] private bool _debugWireframe = false;

    public ConcurrentDictionary<Vector3I, Chunk> VoxelChunks = new();
    private Node3D _player;
    private VoxelJobScheduler _jobScheduler;

    private readonly PriorityQueue<VoxelJob, float> _jobQueue = new();
    private readonly ConcurrentDictionary<Vector3I, VoxelJob> _activeJobs = new();
    private readonly ConcurrentDictionary<Vector3I, CancellationTokenSource> _cancellationTokens = new();

    private const float PRIORITY_TIER_IMMEDIATE_VIEW = 10000.0f;
    private const float PRIORITY_TIER_ANTICIPATION = 5000.0f;
    private const float PRIORITY_TIER_PERIPHERY = 1000.0f;
    private const float PRIORITY_BONUS_MESH = 100.0f;

    private readonly Stopwatch _meshApplicationStopwatch = new();
    private const double MaxMeshApplicationTimeMs = 5.0;

    private readonly FastNoiseLite _terrainNoise = new();
    private readonly FastNoiseLite _caveNoise = new();
    private Vector3I _lastPlayerChunkPosition;
    private bool _isPlayerSpawned = false;

    public override void _Ready()
    {
        _player = GetNode<Node3D>("Player");
        _player.GetNode<CollisionShape3D>("CollisionShape3D").Disabled = true;

        SetupNoise();

        _jobScheduler = new VoxelJobScheduler(this, _terrainNoise, _caveNoise);
        _jobScheduler.Start();

        QueueInitialSpawnArea();
    }

    public override void _Process(double delta)
    {
        if (_isPlayerSpawned)
        {
            Vector3I playerChunkPos = GetPlayerChunkPosition();
            if (playerChunkPos != _lastPlayerChunkPosition)
            {
                _lastPlayerChunkPosition = playerChunkPos;
                UpdateWorldStreamer(playerChunkPos);
            }
        }

        ProcessJobResults();
        DispatchJobs();
        UpdateMetrics();
    }

    public override void _Input(InputEvent e)
    {
        if (e.IsActionPressed("toggle_debug_wireframe"))
            ToggleWireframe();
    }

    private void QueueInitialSpawnArea()
    {
        GD.Print("--- QUEUEING INITIAL SPAWN AREA + BORDER (ASYNC) ---");
        int spawnRadius = 1;
        int loadRadius = spawnRadius + 1;

        for (int x = -loadRadius; x <= loadRadius; x++)
            for (int z = -loadRadius; z <= loadRadius; z++)
                for (int y = -1; y < WorldHeightInChunks + 1; y++)
                {
                    var pos = new Vector3I(x, y, z);
                    var chunk = new Chunk(this, pos);
                    chunk.InitializeMesh(this, VoxelMaterial);
                    VoxelChunks.TryAdd(pos, chunk);
                    EnqueueJob(new VoxelJob(pos, VoxelJobType.GenerateData));
                }
    }

    private void EnqueueJob(VoxelJob job)
    {
        if (_activeJobs.TryAdd(job.Position, job))
        {
            var cts = new CancellationTokenSource();
            job.Token = cts.Token;
            _cancellationTokens.TryAdd(job.Position, cts);
            _jobQueue.Enqueue(job, CalculateJobPriority(job));
        }
    }

    private void DispatchJobs()
    {
        const int maxToDispatch = 8;
        for (int i = 0; i < maxToDispatch && _jobQueue.Count > 0; i++)
            if (_jobQueue.TryDequeue(out VoxelJob job, out _))
            {
                // It's possible the job was cancelled after being dequeued.
                // This check is a safeguard.
                if (job.Token.IsCancellationRequested)
                {
                    CleanupJob(job.Position);
                    continue;
                }
                _jobScheduler.EnqueueJob(job);
            }
    }

    private void ProcessJobResults()
    {
        _meshApplicationStopwatch.Restart();

        while (_jobScheduler.TryDequeueResult(out var result))
        {
            if (_meshApplicationStopwatch.Elapsed.TotalMilliseconds > MaxMeshApplicationTimeMs)
            {
                break;
            }

            var (job, data) = result;

            if (job.Token.IsCancellationRequested)
            {
                CleanupJob(job.Position);
                continue;
            }

            switch (job.JobType)
            {
                case VoxelJobType.GenerateData:
                    foreach (var offset in VoxelUtils.NeighborOffsets)
                    {
                        var neighborPos = job.Position + offset;
                        if (VoxelChunks.ContainsKey(neighborPos))
                        {
                            TryTransitionToMesh(neighborPos);
                        }
                    }
                    TryTransitionToMesh(job.Position);
                    break;

                case VoxelJobType.GenerateMesh:
                    CleanupJob(job.Position);
                    if (VoxelChunks.TryGetValue(job.Position, out var chunk) && data is MeshData meshData)
                    {
                        chunk.ApplyMeshData(meshData);
                        chunk.ApplyCollisionData(meshData);
                    }
                    if (!_isPlayerSpawned)
                        TrySpawnPlayer();
                    break;
            }
        }
    }

    private void TryTransitionToMesh(Vector3I position)
    {
        if (!VoxelChunks.TryGetValue(position, out var chunk) || !chunk.IsDataGenerated)
        {
            return;
        }

        if (!_activeJobs.TryGetValue(position, out var currentJob) ||
            currentJob.JobType != VoxelJobType.GenerateData)
        {
            return;
        }

        foreach (var offset in VoxelUtils.NeighborOffsets)
        {
            var neighbourPos = position + offset;
            if (!VoxelChunks.TryGetValue(neighbourPos, out var neighbour) ||
                !neighbour.IsDataGenerated)
            {
                return;
            }
        }

        CleanupJob(position);
        var meshJob = new VoxelJob(position, VoxelJobType.GenerateMesh);
        EnqueueJob(meshJob);
    }

    private void TrySpawnPlayer()
    {
        if (_isPlayerSpawned)
        {
            GD.Print("--- PLAYER ALREADY SPAWNED! ---"); return;
        }

        for (int y = 0; y < WorldHeightInChunks; y++)
            if (_activeJobs.ContainsKey(new Vector3I(0, y, 0)))
            {
                return;
            }

        _isPlayerSpawned = true;
        GD.Print("--- SPAWN COLUMN READY! SPAWNING PLAYER ---");

        var spawnChunk = VoxelChunks[Vector3I.Zero];
        var spawnPos = FindSafeSpawnPositionInChunk(spawnChunk);

        _player.GlobalPosition = spawnPos + new Vector3(0f, 10f, 0f);
        _player.GetNode<CollisionShape3D>("CollisionShape3D").Disabled = false;

        _lastPlayerChunkPosition = GetPlayerChunkPosition();
        UpdateWorldStreamer(_lastPlayerChunkPosition);
    }

    private Vector3 FindSafeSpawnPositionInChunk(Chunk originChunk)
    {
        const int r = 2;
        int c = Chunk.Size / 2;

        for (int y = Chunk.Size - 3; y >= 0; y--)
            for (int x = -r; x <= r; x++)
                for (int z = -r; z <= r; z++)
                {
                    int vx = c + x, vz = c + z;
                    byte ground = originChunk.GetVoxel(vx, y, vz);
                    if (!VoxelTypes.Definitions[ground].IsSolid) continue;

                    byte air1 = originChunk.GetVoxel(vx, y + 1, vz);
                    byte air2 = originChunk.GetVoxel(vx, y + 2, vz);
                    if (VoxelTypes.Definitions[air1].IsSolid ||
                        VoxelTypes.Definitions[air2].IsSolid) continue;

                    Vector3 w = (Vector3)originChunk.Position * Chunk.Size;
                    return new Vector3(w.X + vx + 0.5f, w.Y + y + 1.0f, w.Z + vz + 0.5f);
                }

        GD.PrintErr("No safe spawn found! Using fallback.");
        return new Vector3(Chunk.Size * 0.5f, Chunk.Size + 10, Chunk.Size * 0.5f);
    }

    private void UpdateWorldStreamer(Vector3I playerChunkPos)
    {
        var required = new HashSet<Vector3I>();
        for (int x = -LoadRadius; x <= LoadRadius; x++)
            for (int z = -LoadRadius; z <= LoadRadius; z++)
                for (int y = -LoadRadius; y < LoadRadius; y++)
                    required.Add(playerChunkPos + new Vector3I(x, y, z));

        // Unload chunks and cancel their jobs
        foreach (var pos in VoxelChunks.Keys.ToArray())
        {
            if (!required.Contains(pos))
            {
                CancelJob(pos);
                if (VoxelChunks.Remove(pos, out var chunk))
                {
                    chunk.Unload();
                }
            }
        }

        // Load new chunks
        foreach (var pos in required)
        {
            if (!VoxelChunks.ContainsKey(pos) && !_activeJobs.ContainsKey(pos))
            {
                var chunk = new Chunk(this, pos);
                chunk.InitializeMesh(this, VoxelMaterial);
                VoxelChunks.TryAdd(pos, chunk);
                EnqueueJob(new VoxelJob(pos, VoxelJobType.GenerateData));
            }
        }
    }

    private void CancelJob(Vector3I pos)
    {
        if (_cancellationTokens.TryRemove(pos, out var cts))
        {
            cts.Cancel();
            cts.Dispose();
        }
        _activeJobs.TryRemove(pos, out _);
    }

    private void CleanupJob(Vector3I pos)
    {
        _activeJobs.TryRemove(pos, out _);
        if (_cancellationTokens.TryRemove(pos, out var cts))
            cts.Dispose();
    }

    private float CalculateJobPriority(VoxelJob job)
    {
        Vector3 center = (Vector3)job.Position * Chunk.Size + Vector3.One * Chunk.Size * 0.5f;
        float dist = _player.GlobalPosition.DistanceTo(center);
        float score = -dist;
        if (dist < LoadRadius * Chunk.Size * 0.5f) score += PRIORITY_TIER_ANTICIPATION;
        if (job.JobType == VoxelJobType.GenerateMesh) score += PRIORITY_BONUS_MESH;

        Vector3 dir = _player.GlobalPosition - _lastPlayerChunkPosition;
        dir.Normalized();
        float dot = dir.Dot(job.Position - _lastPlayerChunkPosition);
        if (dot < 0) score += PRIORITY_TIER_IMMEDIATE_VIEW;

        return score + dist * 1000;
    }

    private Vector3I GetPlayerChunkPosition() =>
        new((int)Mathf.Floor(_player.GlobalPosition.X / Chunk.Size),
            (int)Mathf.Floor(_player.GlobalPosition.Y / Chunk.Size),
            (int)Mathf.Floor(_player.GlobalPosition.Z / Chunk.Size));

    private void SetupNoise()
    {
        _terrainNoise.Seed = (int)GD.Randi();
        _terrainNoise.Frequency = 0.005f;
        _caveNoise.Seed = (int)GD.Randi();
        _caveNoise.Frequency = 0.02f;
        _caveNoise.FractalType = FastNoiseLite.FractalTypeEnum.Ridged;
    }

    private void UpdateMetrics()
    {
        if (PerformanceLabel == null) return;

        double fps = Performance.GetMonitor(Performance.Monitor.TimeFps);
        long verts = (long)Performance.GetMonitor(Performance.Monitor.RenderTotalPrimitivesInFrame);
        long draws = (long)Performance.GetMonitor(Performance.Monitor.RenderTotalDrawCallsInFrame);
        long queued = _jobQueue.Count;
        long active = _activeJobs.Count;
        Vector3 playerPos = _player.GlobalPosition;

        PerformanceLabel.Text =
            $"FPS: {fps:F0}\n" +
            $"Vertices: {verts}\n" +
            $"Draws: {draws}\n" +
            $"Queued: {queued}\n" +
            $"Active: {active}" +
            $"\nPlayer: {playerPos}";
    }

    private void ToggleWireframe()
    {
        _debugWireframe = !_debugWireframe;
        GD.Print("Wireframe: " + _debugWireframe);

        var vp = GetViewport();
        vp.DebugDraw = _debugWireframe ? Viewport.DebugDrawEnum.Wireframe : Viewport.DebugDrawEnum.Disabled;

        foreach (var child in GetChildren())
            if (child is MeshInstance3D mi && mi.MaterialOverride is StandardMaterial3D mat)
                mat.CullMode = _debugWireframe
                    ? BaseMaterial3D.CullModeEnum.Disabled
                    : BaseMaterial3D.CullModeEnum.Back;
    }
}
