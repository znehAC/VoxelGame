using Godot;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;

public enum VoxelJobType { GenerateData, GenerateMesh }

public struct VoxelJob
{
    public readonly Vector3I Position;
    public readonly VoxelJobType JobType;
    public CancellationToken Token;

    public VoxelJob(Vector3I position, VoxelJobType jobType)
    {
        Position = position;
        JobType = jobType;
        Token = CancellationToken.None;
    }
}

public class VoxelJobScheduler
{
    public int WorkerCount { get; private set; }
    public int WorkQueueCount => _workQueue.Count;

    private readonly World _world;
    private readonly FastNoiseLite _terrainNoise;
    private readonly FastNoiseLite _caveNoise;

    private readonly BlockingCollection<VoxelJob> _workQueue = new();
    private readonly ConcurrentQueue<(VoxelJob job, object data)> _resultsQueue = new();

    public VoxelJobScheduler(World world, FastNoiseLite terrainNoise, FastNoiseLite caveNoise)
    {
        _world = world;
        _terrainNoise = terrainNoise;
        _caveNoise = caveNoise;
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
                            if (!chunk.IsDataGenerated)
                            {
                                chunk.PrepareData(_terrainNoise, _caveNoise, job.Token);
                            }
                            break;

                        case VoxelJobType.GenerateMesh:
                            resultData = chunk.GenerateMesh(job.Token);
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
            }
            catch (System.Exception e)
            {
                GD.PrintErr($"[WORKER-{workerId}] ERROR on {job.Position}: {e.Message}");
                GD.PrintErr($"[WORKER-{workerId}] Stack trace: {e.StackTrace}");
            }
        }
    }

    public void Stop()
    {
        _workQueue.CompleteAdding();
    }
}
