using Godot;
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
        CenterVoxels = centerChunk.GetVoxels();
        NeighborVoxels = new byte[6][];
        for (int i = 0; i < 6; i++)
        {
            if (neighbors[i] != null)
            {
                NeighborVoxels[i] = neighbors[i].GetVoxels();
            }
        }
    }
}


public struct VoxelJob
{
    public readonly Vector3I Position;
    public readonly VoxelJobType JobType;
    public CancellationToken Token;
    public readonly object Payload;

    public VoxelJob(Vector3I position, VoxelJobType jobType, CancellationToken token, object payload = null)
    {
        Position = position;
        JobType = jobType;
        Token = token;
        Payload = payload;
    }
}

public class VoxelJobScheduler
{
    public int WorkerCount { get; private set; }
    public int WorkQueueCount => _workQueue.Count;

    private readonly World _world;
    private readonly FastNoiseLite _continentalnessNoise;
    private readonly FastNoiseLite _erosionNoise;
    private readonly FastNoiseLite _caveNoise;

    private readonly BlockingCollection<VoxelJob> _workQueue = new();
    private readonly ConcurrentQueue<(VoxelJob job, object data)> _resultsQueue = new();


    private readonly BaseMesher _mesher;

    public VoxelJobScheduler(World world, FastNoiseLite continentalnessNoise, FastNoiseLite erosionNoise, FastNoiseLite caveNoise, BaseMesher mesher)
    {
        _world = world;
        _continentalnessNoise = continentalnessNoise;
        _erosionNoise = erosionNoise;
        _caveNoise = caveNoise;
        _mesher = mesher;
    }

    public void Start()
    {
        WorkerCount = System.Math.Max(1, System.Environment.ProcessorCount - 2);
        GD.Print($"Starting {WorkerCount} voxel job workers.");

        for (int i = 0; i < WorkerCount; i++)
        {
            int workerId = i;
            Task.Run(() => WorkerLoop(workerId));
        }
    }

    public void EnqueueJob(VoxelJob job)
    {
        _workQueue.Add(job);
    }

    public bool TryDequeueResult(out (VoxelJob job, object data) result)
    {
        return _resultsQueue.TryDequeue(out result);
    }

    private void WorkerLoop(int workerId)
    {
        GD.Print($"[WORKER-{workerId}] Starting worker loop.");
        foreach (var job in _workQueue.GetConsumingEnumerable())
        {
            try
            {
                job.Token.ThrowIfCancellationRequested();

                if (_world.VoxelChunks.TryGetValue(job.Position, out Chunk chunk))
                {
                    object resultData = null;

                    switch (job.JobType)
                    {
                        case VoxelJobType.GenerateData:
                            chunk.PrepareData(_continentalnessNoise, _erosionNoise, _caveNoise, job.Token);
                            break;

                        case VoxelJobType.GenerateMesh:
                            if (job.Payload is MeshJobData meshJobData)
                            {
                                resultData = _mesher.GenerateMeshData(meshJobData, VoxelTypes.Definitions, job.Token);
                            }
                            else
                            {
                                GD.PrintErr($"[WORKER-{workerId}] ERROR: GenerateMesh job for {job.Position} received invalid payload.");
                            }
                            break;
                    }

                    job.Token.ThrowIfCancellationRequested();
                    _resultsQueue.Enqueue((job, resultData));
                }
                else
                {
                    GD.PrintErr($"[WORKER-{workerId}] Chunk not found: {job.Position}");
                }

            }
            catch (System.OperationCanceledException)
            {
                GD.Print($"[WORKER-{workerId}] Job for chunk {job.Position} was cancelled.");
            }
            catch (System.Exception e)
            {
                GD.PrintErr($"[WORKER-{workerId}] ERROR on {job.Position}: {e.Message}");
                GD.PrintErr($"[WORK-ER-{workerId}] Stack trace: {e.StackTrace}");
            }
        }
        GD.Print($"[WORKER-{workerId}] Exiting worker loop.");
    }

    public void Stop()
    {
        _workQueue.CompleteAdding();
    }
}
