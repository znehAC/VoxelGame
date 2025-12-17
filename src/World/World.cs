using Godot;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using VoxelPowderSim.Src.Voxel.Data;

public partial class World : Node3D
{
    // Use Resource type to prevent crash if user assigns a Script file
    [Export] public Resource Config { get; set; }
    public TerrainConfig ActiveConfig { get; private set; }

    [Export] public StandardMaterial3D VoxelMaterial { get; set; }
    
    // Kept for debug/metrics, logic moved to Config where possible
    [Export] private Label PerformanceLabel { get; set; }
    [Export] private bool _debugWireframe = false;
    
    public int Seed { get; private set; }

    // Dictionary is now just a Lookup Cache for Mesher/Physics.
    public readonly ConcurrentDictionary<Vector4I, Chunk> VoxelChunks = new();

    private Node3D _player;
    private Camera3D _camera;
    private VoxelJobScheduler _jobScheduler;
    private BaseMesher _mesher;
    private ChunkPool _chunkPool;
    
    private ChunkOctree _octree;
    private bool _isPlayerSpawned = false;
    
    private double _worldStreamerTimer = 0.0;
    private const double WorldStreamerInterval = 0.05; // 20 Hz
            
    private readonly Queue<Chunk> _chunksAwaitingData = new();
    private readonly Queue<Chunk> _chunksAwaitingMesh = new();

    // Phase 2: Floating Origin Accumulator
    private Vector3 _worldOriginOffset = Vector3.Zero;
    public Vector3 WorldOriginOffset => _worldOriginOffset;

    public override void _Ready()
    {
        // 1. Load or Validate Config
        if (Config is TerrainConfig tc)
        {
            ActiveConfig = tc;
        }
        else
        {
            if (Config != null)
            {
                GD.PrintErr($"[World] Assigned Config was not a TerrainConfig instance! (Type: {Config.GetType().Name}). Using Default.");
            }
            ActiveConfig = new TerrainConfig();
        }

        if (PerformanceLabel != null)
        {
             PerformanceLabel.AutowrapMode = TextServer.AutowrapMode.Word;
        }

        _player = GetNode<Node3D>("Player");
        _camera = _player.GetNode<Camera3D>("Camera3D");
        _camera.Far = ActiveConfig.OriginShiftThreshold * 1.5f; // Ensure frustum covers shift distance
        
        _player.GetNode<CollisionShape3D>("CollisionShape3D").Disabled = true;

        if (VoxelMaterial == null) VoxelMaterial = new StandardMaterial3D();
        VoxelMaterial.VertexColorUseAsAlbedo = true;
        VoxelMaterial.CullMode = BaseMaterial3D.CullModeEnum.Disabled;
        VoxelMaterial.ShadingMode = BaseMaterial3D.ShadingModeEnum.Unshaded;
        
        // Ensure Albedo Color is White so it doesn't tint vertex colors
        VoxelMaterial.AlbedoColor = Colors.White;

        GD.Print($"[Material Config] VertexColor: {VoxelMaterial.VertexColorUseAsAlbedo}, Cull: {VoxelMaterial.CullMode}");

        // --- DEBUG RED CUBE ---
        var debugCube = new MeshInstance3D();
        debugCube.Mesh = new BoxMesh { Size = Vector3.One * 5.0f };
        var redMat = new StandardMaterial3D { AlbedoColor = Colors.Red, ShadingMode = BaseMaterial3D.ShadingModeEnum.Unshaded };
        debugCube.MaterialOverride = redMat;
        AddChild(debugCube);
        debugCube.GlobalPosition = new Vector3(0, 105, -10); // In front of player spawn
        GD.Print("Created Debug Red Cube at (0, 105, -10)");
        // ---------------------

        _chunkPool = new ChunkPool(this);
        Seed = (int)GD.Randi();

        _mesher = new GreedyMesher();
        
        _jobScheduler = new VoxelJobScheduler(this, _mesher, Seed);
        _jobScheduler.Start();
        
        // Initialize Octree (Used for Fixed Mode)
        _octree = new ChunkOctree(this, _chunkPool);

        GD.Print($"World Initialized. Mode: {ActiveConfig.Mode}, Scale: {ActiveConfig.VoxelScale}");
    }

    public override void _Process(double delta)
    {
        _worldStreamerTimer += delta;
        
        // Phase 2: Infinite Scrolling & Origin Shifting
        if (ActiveConfig.Mode == TerrainConfig.WorldMode.Infinite)
        {
            CheckFloatingOrigin();
            
            if (_worldStreamerTimer >= WorldStreamerInterval)
            {
                // Unified Octree Logic
                Vector3 logicalPos = _player.GlobalPosition + _worldOriginOffset;
                _octree.Update(logicalPos);
                _worldStreamerTimer = 0.0;
            }
        }
        else // Fixed Mode
        {
            if (_worldStreamerTimer >= WorldStreamerInterval)
            {
                // Fixed mode assumes origin at 0, no shift
                _octree.Update(_player.GlobalPosition);
                _worldStreamerTimer = 0.0;
            }
        }

        ProcessJobResults();
        ProcessStateTransitions();
        UpdateMetrics();
    }

    private void CheckFloatingOrigin()
    {
        float dist = _player.Position.Length();
        if (dist > ActiveConfig.OriginShiftThreshold)
        {
            Vector3 shift = -_player.Position;
            ApplyWorldShift(shift);
        }
    }

    private void ApplyWorldShift(Vector3 shift)
    {
        // 1. Shift Player (Reset to near zero)
        _player.Position += shift;

        // 2. Accumulate Shift (for Logical <-> Physical conversion if needed later)
        _worldOriginOffset -= shift;

        // 3. Shift Active Chunks
        // Note: Chunk.Position is the LOGICAL coordinate (Index).
        // We shift the VISUAL representation (MeshInstance).
        foreach (var kvp in VoxelChunks)
        {
            Chunk chunk = kvp.Value;
            if (GodotObject.IsInstanceValid(chunk.MeshInstance))
            {
                chunk.MeshInstance.Position += shift;
            }
        }

        GD.Print($"[World] Floating Origin Shift: {shift}. Total Offset: {_worldOriginOffset}");
    }

    // Called by Octree or Infinite Loop
    public void RequestChunkLoad(Chunk chunk)
    {
        if (chunk.State == ChunkState.AwaitingData)
        {
            chunk.SetVisible(false);
            if (!_chunksAwaitingData.Contains(chunk))
            {
                _chunksAwaitingData.Enqueue(chunk);
            }
        }
    }

    public override void _Input(InputEvent e)
    {
        if (e.IsActionPressed("toggle_debug_wireframe"))
        {
            ToggleWireframe();
        }
    }

    private void ToggleWireframe()
    {
        _debugWireframe = !_debugWireframe;
        GetViewport().DebugDraw = _debugWireframe ? Viewport.DebugDrawEnum.Wireframe : Viewport.DebugDrawEnum.Disabled;
    }

    private void ProcessJobResults()
    {
        const int MaxMeshesPerFrame = 64;
        int meshesApplied = 0;

        while (meshesApplied < MaxMeshesPerFrame && _jobScheduler.TryDequeueResult(out var result))
        {
            var (job, data) = result;
            var key = new Vector4I(job.Position.X, job.Position.Y, job.Position.Z, job.Lod);
            
            if (VoxelChunks.TryGetValue(key, out var chunk))
            {
                switch (job.JobType)
                {
                    case VoxelJobType.GenerateData:
                        if (data is (byte[] voxels, bool isEmpty))
                        {
                            chunk.SetVoxels(voxels, isEmpty);
                            
                            if (chunk.IsEmpty) 
                            {
                                chunk.State = ChunkState.Ready;
                            }
                            else
                            {
                                chunk.State = ChunkState.AwaitingMesh;
                                _chunksAwaitingMesh.Enqueue(chunk);
                            }
                        }
                        break;
                    case VoxelJobType.GenerateMesh when data is MeshData meshData:
                        chunk.ApplyMeshData(meshData);
                        if (chunk.Lod == 0) chunk.ApplyCollisionData(meshData);
                        chunk.State = ChunkState.Ready;
                        
                        // Force visibility ON for Infinite Mode (or let Octree handle it in Fixed)
                        // Ideally, we just turn it on. Octree will hide it if needed next frame.
                        chunk.SetVisible(true);
                        
                        if (chunk.Lod == 0 && chunk.Position == new Vector3I(0, 1, 0))
                        {
                             TrySpawnPlayer();
                        }
                        meshesApplied++;
                        break;
                }
            }
        }
    }

    private void ProcessStateTransitions()
    {
        const int ChunksToProcessPerFrame = 128;

        // 1. Data Generation
        int dataQueueCount = _chunksAwaitingData.Count;
        for (int i = 0; i < dataQueueCount && i < ChunksToProcessPerFrame; i++)
        {
            if (_chunksAwaitingData.TryDequeue(out var chunk))
            {
                var key = new Vector4I(chunk.Position.X, chunk.Position.Y, chunk.Position.Z, chunk.Lod);
                if (VoxelChunks.ContainsKey(key) && chunk.State == ChunkState.AwaitingData)
                {
                    chunk.State = ChunkState.GeneratingData;
                    _jobScheduler.EnqueueJob(new VoxelJob(chunk.Position, VoxelJobType.GenerateData, chunk.JobCancellationTokenSource.Token, chunk.Lod));
                }
            }
        }

        // 2. Mesh Generation
        int meshQueueCount = _chunksAwaitingMesh.Count;
        for (int i = 0; i < meshQueueCount && i < ChunksToProcessPerFrame; i++)
        {
            if (!_chunksAwaitingMesh.TryDequeue(out var chunk)) continue;

             var key = new Vector4I(chunk.Position.X, chunk.Position.Y, chunk.Position.Z, chunk.Lod);
             if (!VoxelChunks.ContainsKey(key)) continue; 

            if (chunk.State == ChunkState.AwaitingMesh)
            {
                var neighbors = new Chunk[6];
                bool allNeighborsReady = true;

                for (int j = 0; j < VoxelUtils.NeighborOffsets.Length; j++)
                {
                    var neighborPos = chunk.Position + VoxelUtils.NeighborOffsets[j];
                    var neighborKey = new Vector4I(neighborPos.X, neighborPos.Y, neighborPos.Z, chunk.Lod);
                    
                    if (VoxelChunks.TryGetValue(neighborKey, out var neighbor))
                    {
                        if (neighbor.State >= ChunkState.AwaitingMesh) 
                        {
                            neighbors[j] = neighbor;
                        }
                        else 
                        {
                            allNeighborsReady = false; 
                            break; 
                        }
                    }
                    else 
                    {
                        neighbors[j] = null;
                    }
                }

                if (allNeighborsReady)
                {
                    chunk.State = ChunkState.Meshing;
                    var meshJobData = new MeshJobData(chunk, neighbors);
                    _jobScheduler.EnqueueJob(new VoxelJob(chunk.Position, VoxelJobType.GenerateMesh, chunk.JobCancellationTokenSource.Token, chunk.Lod, meshJobData));
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
        
        var shape = _player.GetNode<CollisionShape3D>("CollisionShape3D");
        if (shape.Disabled)
        {
            GD.Print("--- SPAWN CHUNK READY! TELEPORTING & ENABLING PLAYER ---");
            // Spawn high to avoid falling through terrain, but respect scale
            // Old: 100 units. New: 100 * Scale (e.g. 10m)
            float spawnY = 100.0f * ActiveConfig.VoxelScale;
            _player.GlobalPosition = new Vector3(0, spawnY, 0);
            shape.Disabled = false;
            _isPlayerSpawned = true;
        }
    }
    
    private void UpdateMetrics()
    {
        if (PerformanceLabel == null) return;
        double fps = Performance.GetMonitor(Performance.Monitor.TimeFps);
        long verts = (long)Performance.GetMonitor(Performance.Monitor.RenderTotalPrimitivesInFrame);
        double mem = Process.GetCurrentProcess().PrivateMemorySize64 / (1024.0 * 1024.0);
        
        int loadedChunks = VoxelChunks.Count;
        int readyChunks = 0;
        foreach (var c in VoxelChunks.Values) if (c.State == ChunkState.Ready) readyChunks++;
        
        Vector3 p = _player.GlobalPosition;
        // Show Real + Virtual Position
        Vector3 logicalPos = p + _worldOriginOffset;

        PerformanceLabel.Text = $"FPS: {fps:F0} | Verts: {verts} | Mem: {mem:F0} MB | Chunks: {loadedChunks} ({readyChunks} Ready)\nVisPos: {p.X:F1}, {p.Y:F1}, {p.Z:F1}\nLogPos: {logicalPos.X:F1}, {logicalPos.Y:F1}, {logicalPos.Z:F1}";
    }
}