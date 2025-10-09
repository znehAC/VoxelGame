using Godot;
using System.Collections.Generic;
using System.Collections.Concurrent;

public partial class World : Node3D
{
    [Export] public StandardMaterial3D VoxelMaterial { get; set; }
    [Export] public int LoadRadius { get; set; } = 8;
    [Export] public int WorldHeightInChunks { get; set; } = 10;
    [Export] private Label PerformanceLabel { get; set; }
    [Export] private bool _debugWireframe = false;




    [Export] public int MaxChunksToLoadPerFrame { get; set; } = 8;

    public readonly ConcurrentDictionary<Vector3I, Chunk> VoxelChunks = new();
    private Node3D _player;
    private Camera3D _camera;
    private VoxelJobScheduler _jobScheduler;
    private BaseMesher _mesher;
    private Vector3I _lastPlayerChunkPosition;
    private bool _isPlayerSpawned = false;
    private ChunkPool _chunkPool;

    private double _worldStreamerTimer = 0.0;
    private const double WorldStreamerInterval = 0.1;

    private readonly List<Vector3I> _chunksToUnload = new();

    private readonly Queue<Chunk> _chunksAwaitingData = new();
    private readonly Queue<Chunk> _chunksAwaitingMesh = new();

    private readonly FastNoiseLite _continentalnessNoise = new();
    private readonly FastNoiseLite _erosionNoise = new();
    private readonly FastNoiseLite _caveNoise = new();

    public override void _Ready()
    {
        _player = GetNode<Node3D>("Player");
        _camera = _player.GetNode<Camera3D>("Camera3D");
        _player.GetNode<CollisionShape3D>("CollisionShape3D").Disabled = true;
        _lastPlayerChunkPosition = GetPlayerChunkPosition();

        _chunkPool = new ChunkPool(this);
        SetupNoise();


        _mesher = new CulledMesher();
        _jobScheduler = new VoxelJobScheduler(this, _continentalnessNoise, _erosionNoise, _caveNoise, _mesher);
        _jobScheduler.Start();


    }

    public override void _Process(double delta)
    {
        _worldStreamerTimer += delta;
        if (_worldStreamerTimer >= WorldStreamerInterval)
        {
            Vector3I playerChunkPos = GetPlayerChunkPosition();


            UpdateWorldStreamer(playerChunkPos);
            _lastPlayerChunkPosition = playerChunkPos;

            _worldStreamerTimer = 0.0;
        }

        ProcessJobResults();
        ProcessStateTransitions();
        UpdateMetrics();
    }

    public override void _Input(InputEvent e)
    {
        if (e.IsActionPressed("toggle_debug_wireframe"))
        {
            ToggleWireframe();
        }
    }



    private void UpdateWorldStreamer(Vector3I playerChunkPos)
    {
        var requiredPositions = new HashSet<Vector3I>();
        var chunksToLoad = new List<(Vector3I pos, float priority)>();
        var playerForward = -_camera.GlobalTransform.Basis.Z;


        for (int x = -LoadRadius; x <= LoadRadius; x++)
        {
            for (int z = -LoadRadius; z <= LoadRadius; z++)
            {

                for (int y = LoadRadius; y >= -LoadRadius; y--)
                {
                    var chunkPos = playerChunkPos + new Vector3I(x, y, z);
                    requiredPositions.Add(chunkPos);


                    if (VoxelChunks.TryGetValue(chunkPos, out var chunk) && chunk.IsFullyOpaque)
                    {


                        break;
                    }

                    if (!VoxelChunks.ContainsKey(chunkPos))
                    {

                        var chunkWorldCenter = (Vector3)chunkPos * Chunk.Size + (Vector3.One * (Chunk.Size / 2f));
                        var directionToChunk = (chunkWorldCenter - _player.GlobalPosition).Normalized();
                        var distance = playerChunkPos.DistanceTo(chunkPos);

                        float dot = playerForward.Dot(directionToChunk);
                        float priority = (1.0f / Mathf.Max(1.0f, distance)) * (dot + 1.1f);

                        chunksToLoad.Add((chunkPos, priority));
                    }
                }
            }
        }


        _chunksToUnload.Clear();
        foreach (var pos in VoxelChunks.Keys)
        {
            if (!requiredPositions.Contains(pos))
            {
                _chunksToUnload.Add(pos);
            }
        }

        foreach (var pos in _chunksToUnload)
        {
            if (VoxelChunks.TryRemove(pos, out var chunk))
            {
                chunk.Unload();
                _chunkPool.Return(chunk);
            }
        }



        chunksToLoad.Sort((a, b) => b.priority.CompareTo(a.priority));


        int chunksLoadedThisFrame = 0;
        foreach (var (pos, _) in chunksToLoad)
        {
            if (chunksLoadedThisFrame >= MaxChunksToLoadPerFrame)
            {
                break;
            }


            if (VoxelChunks.ContainsKey(pos)) continue;

            var newChunk = _chunkPool.Get(pos);
            newChunk.InitializeMeshNode(this, VoxelMaterial);
            if (VoxelChunks.TryAdd(pos, newChunk))
            {
                _chunksAwaitingData.Enqueue(newChunk);
                chunksLoadedThisFrame++;
            }
        }
    }


    private void ProcessJobResults()
    {
        const int MaxMeshesPerFrame = 8;
        int meshesApplied = 0;

        while (meshesApplied < MaxMeshesPerFrame && _jobScheduler.TryDequeueResult(out var result))
        {
            var (job, data) = result;
            if (VoxelChunks.TryGetValue(job.Position, out var chunk))
            {

                switch (job.JobType)
                {
                    case VoxelJobType.GenerateData:
                        chunk.State = ChunkState.AwaitingMesh;
                        _chunksAwaitingMesh.Enqueue(chunk);
                        break;
                    case VoxelJobType.GenerateMesh when data is MeshData meshData:
                        chunk.ApplyMeshData(meshData);
                        chunk.State = ChunkState.Ready;
                        if (!_isPlayerSpawned) TrySpawnPlayer();
                        meshesApplied++;
                        break;
                }
            }
        }
    }

    private void ProcessStateTransitions()
    {
        const int ChunksToProcessPerFrame = 64;


        int dataQueueCount = _chunksAwaitingData.Count;
        for (int i = 0; i < dataQueueCount && i < ChunksToProcessPerFrame; i++)
        {
            if (_chunksAwaitingData.TryDequeue(out var chunk))
            {
                if (chunk.State == ChunkState.AwaitingData)
                {
                    chunk.State = ChunkState.GeneratingData;

                    _jobScheduler.EnqueueJob(new VoxelJob(chunk.Position, VoxelJobType.GenerateData, chunk.JobCancellationTokenSource.Token));
                }

            }
        }


        int meshQueueCount = _chunksAwaitingMesh.Count;
        for (int i = 0; i < meshQueueCount && i < ChunksToProcessPerFrame; i++)
        {
            if (!_chunksAwaitingMesh.TryDequeue(out var chunk)) continue;

            if (chunk.State == ChunkState.AwaitingMesh)
            {
                var neighbors = new Chunk[6];
                bool allNeighborsReady = true;

                for (int j = 0; j < VoxelUtils.NeighborOffsets.Length; j++)
                {
                    var neighborPos = chunk.Position + VoxelUtils.NeighborOffsets[j];
                    if (VoxelChunks.TryGetValue(neighborPos, out var neighbor) && neighbor.State >= ChunkState.AwaitingMesh)
                    {
                        neighbors[j] = neighbor;
                    }
                    else
                    {
                        allNeighborsReady = false;
                        break;
                    }
                }

                if (allNeighborsReady)
                {
                    chunk.State = ChunkState.Meshing;

                    var meshJobData = new MeshJobData(chunk, neighbors);
                    _jobScheduler.EnqueueJob(new VoxelJob(chunk.Position, VoxelJobType.GenerateMesh, chunk.JobCancellationTokenSource.Token, meshJobData));
                }
                else
                {
                    _chunksAwaitingMesh.Enqueue(chunk);
                }
            }
        }
    }


    private void TrySpawnPlayer()
    {
        if (_isPlayerSpawned) return;

        if (VoxelChunks.TryGetValue(Vector3I.Zero, out var chunk) && chunk.State == ChunkState.Ready)
        {
            _isPlayerSpawned = true;
            GD.Print("--- SPAWN CHUNK READY! ENABLING PLAYER ---");
            _player.GetNode<CollisionShape3D>("CollisionShape3D").Disabled = false;
        }
    }
    private Vector3I GetPlayerChunkPosition() =>
        new((int)Mathf.Floor(_player.GlobalPosition.X / Chunk.Size),
            (int)Mathf.Floor(_player.GlobalPosition.Y / Chunk.Size),
            (int)Mathf.Floor(_player.GlobalPosition.Z / Chunk.Size));

    private void SetupNoise()
    {
        var seed = (int)GD.Randi();

        _continentalnessNoise.Seed = seed;
        _continentalnessNoise.Frequency = 0.002f;
        _continentalnessNoise.FractalType = FastNoiseLite.FractalTypeEnum.Fbm;

        _erosionNoise.Seed = seed + 1;
        _erosionNoise.Frequency = 0.008f;
        _erosionNoise.FractalType = FastNoiseLite.FractalTypeEnum.Ridged;

        _caveNoise.Seed = seed + 2;
        _caveNoise.Frequency = 0.02f;
        _caveNoise.FractalType = FastNoiseLite.FractalTypeEnum.Ridged;
    }

    private void UpdateMetrics()
    {
        if (PerformanceLabel == null) return;
        double fps = Performance.GetMonitor(Performance.Monitor.TimeFps);
        long verts = (long)Performance.GetMonitor(Performance.Monitor.RenderTotalPrimitivesInFrame);
        PerformanceLabel.Text = $"FPS: {fps:F0}\nVertices: {verts}\nChunks: {VoxelChunks.Count}";
    }

    private void ToggleWireframe()
    {
        _debugWireframe = !_debugWireframe;
        GD.Print("Wireframe: " + _debugWireframe);

        var vp = GetViewport();
        vp.DebugDraw = _debugWireframe ? Viewport.DebugDrawEnum.Wireframe : Viewport.DebugDrawEnum.Disabled;
    }
}
