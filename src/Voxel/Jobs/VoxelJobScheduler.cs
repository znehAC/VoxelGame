using Godot;
using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;

public enum VoxelJobType { GenerateData, GenerateMesh }

public class MeshJobData
{
    public byte[] CenterVoxels { get; }
    public byte[][] NeighborVoxels { get; }

    public MeshJobData(Chunk centerChunk, Chunk[] neighbors)
    {
        // COPY the data to avoid race conditions (Zombie Chunks)
        int len = Chunk.Size * Chunk.Size * Chunk.Size;
        
        CenterVoxels = new byte[len];
        Buffer.BlockCopy(centerChunk.GetVoxels(), 0, CenterVoxels, 0, len);

        NeighborVoxels = new byte[6][];
        for (int i = 0; i < 6; i++)
        {
            if (neighbors[i] != null)
            {
                NeighborVoxels[i] = new byte[len];
                Buffer.BlockCopy(neighbors[i].GetVoxels(), 0, NeighborVoxels[i], 0, len);
            }
        }
    }
}

public struct VoxelJob
{
    public readonly Vector3I Position;
    public readonly int Lod;
    public readonly VoxelJobType JobType;
    public CancellationToken Token;
    public readonly object Payload;

    public VoxelJob(Vector3I position, VoxelJobType jobType, CancellationToken token, int lod = 0, object payload = null)
    {
        Position = position;
        JobType = jobType;
        Token = token;
        Lod = lod;
        Payload = payload;
    }
}

public class VoxelJobScheduler
{
    public int WorkQueueCount => _meshQueue.Count + _computeQueue.Count;

    private readonly World _world;
    private readonly BlockingCollection<VoxelJob> _computeQueue = new();
    private readonly BlockingCollection<VoxelJob> _meshQueue = new();
    private readonly ConcurrentQueue<(VoxelJob job, object data)> _resultsQueue = new();

    private readonly BaseMesher _mesher;
    private readonly int _seed;
    private bool _running = false;
    private CancellationTokenSource _schedulerCts;

    public VoxelJobScheduler(World world, BaseMesher mesher, int seed)
    {
        _world = world;
        _mesher = mesher;
        _seed = seed;
    }

    public void Start()
    {
        if (_running) return;
        _running = true;
        _schedulerCts = new CancellationTokenSource();

        // Pre-load shader source to avoid I/O on every thread
        string shaderSource = "";
        using (var file = Godot.FileAccess.Open("res://src/Voxel/Shaders/TerrainGenerator.glsl", Godot.FileAccess.ModeFlags.Read))
        {
            if (file != null) shaderSource = file.GetAsText();
            else GD.PrintErr("VoxelJobScheduler: Could not open shader source file.");
        }

        int totalThreads = System.Math.Max(2, System.Environment.ProcessorCount - 2);
        int computeThreads = 2; 
        int meshThreads = System.Math.Max(1, totalThreads - computeThreads);

        GD.Print($"Starting Voxel Scheduler: {computeThreads} Compute Threads, {meshThreads} Mesh Threads.");
        
        // Spawn Compute Workers (Local RD)
        for (int i = 0; i < computeThreads; i++)
        {
            int id = i;
            string sourceCapture = shaderSource;
            Task.Run(() => ComputeLoop(id, sourceCapture, _schedulerCts.Token));
        }

        // Spawn Mesh Workers (CPU)
        for (int i = 0; i < meshThreads; i++)
        {
            int id = i + computeThreads;
            Task.Run(() => MeshLoop(id, _schedulerCts.Token));
        }
    }

    public void EnqueueJob(VoxelJob job)
    {
        if (!_running) return;
        if (job.JobType == VoxelJobType.GenerateData)
            _computeQueue.Add(job);
        else
            _meshQueue.Add(job);
    }

    public bool TryDequeueResult(out (VoxelJob job, object data) result)
    {
        return _resultsQueue.TryDequeue(out result);
    }

    private void ComputeLoop(int workerId, string shaderSource, CancellationToken schedulerToken)
    {
        GD.Print($"[COMPUTE-{workerId}] Starting Compute Loop.");
        using (var terrainCompute = new TerrainCompute())
        {
            terrainCompute.Initialize(shaderSource);

            try 
            {
                foreach (var job in _computeQueue.GetConsumingEnumerable(schedulerToken))
                {
                    if (job.Token.IsCancellationRequested) continue;

                    // Note: We don't check World.VoxelChunks here anymore for the *Chunk object* 
                    // because we might be in a different state, but for 'GenerateData' we write BACK to the chunk.
                    // The job holds Position/Lod.
                    // However, 'GenerateData' produces a byte array. 
                    // We shouldn't write to the chunk directly from this thread if we want strict safety, 
                    // but Chunk.SetVoxels uses a lock or is atomic enough? No.
                    // Ideally, we return the data to Main Thread.
                    // BUT, current architecture writes to chunk.
                    // Octree/Dictionary holds the chunk.
                    // If the chunk was unloaded, it might be returned to pool.
                    // Checking if valid?
                    // The safer way: Return the byte[] to the main thread.
                    
                    try
                    {
                         var (data, isEmpty) = terrainCompute.Generate(job.Position, _seed, 1 << job.Lod, job.Lod);
                         // Return data, don't modify Chunk directly on thread
                         _resultsQueue.Enqueue((job, (data, isEmpty))); 
                    }
                    catch (Exception e)
                    {
                        GD.PrintErr($"[COMPUTE-{workerId}] Error: {e.Message}");
                    }
                }
            }
            catch (OperationCanceledException) { }
        }
    }

    private void MeshLoop(int workerId, CancellationToken schedulerToken)
    {
        GD.Print($"[MESH-{workerId}] Starting Mesh Loop.");
        try
        {
            foreach (var job in _meshQueue.GetConsumingEnumerable(schedulerToken))
            {
                if (job.Token.IsCancellationRequested) continue;

                try
                {
                    // Job payload already contains the data snapshot (MeshJobData).
                    // No need to look up Chunk in World.Dictionary.
                    
                    if (job.JobType == VoxelJobType.GenerateMesh && job.Payload is MeshJobData meshJobData)
                    {
                         var resultData = _mesher.GenerateMeshData(meshJobData, VoxelTypes.Definitions, job.Token);
                         _resultsQueue.Enqueue((job, resultData));
                    }
                }
                catch (Exception e)
                {
                     GD.PrintErr($"[MESH-{workerId}] Error: {e.Message}");
                }
            }
        }
        catch (OperationCanceledException) { }
    }

    public void Stop()
    {
        _running = false;
        _schedulerCts?.Cancel();
        _computeQueue.CompleteAdding();
        _meshQueue.CompleteAdding();
    }
}