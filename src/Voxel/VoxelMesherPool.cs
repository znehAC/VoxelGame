using Godot;
using System.Collections.Concurrent;
using System.Threading.Tasks;

public enum VoxelJobType { GenerateData, GenerateMesh }

public readonly struct VoxelJob
{
    public readonly Vector3I Position;
    public readonly VoxelJobType JobType;

    public VoxelJob(Vector3I position, VoxelJobType jobType)
    {
        Position = position;
        JobType = jobType;
    }
}

public class VoxelMesherPool
{
    public int WorkerCount { get; private set; }
    public int WorkQueueCount => _workQueue.Count;

    private readonly World _world;
    private readonly FastNoiseLite _terrainNoise;
    private readonly FastNoiseLite _caveNoise;

    private readonly ConcurrentQueue<VoxelJob> _workQueue = new();
    private readonly ConcurrentQueue<(VoxelJob job, object data)> _resultsQueue = new();

    public VoxelMesherPool(World world, FastNoiseLite terrainNoise, FastNoiseLite caveNoise)
    {
        _world = world;
        _terrainNoise = terrainNoise;
        _caveNoise = caveNoise;
    }

    public void Start()
    {
        WorkerCount = System.Math.Max(1, System.Environment.ProcessorCount / 2);
        GD.Print($"Starting {WorkerCount} meshing workers.");

        for (int i = 0; i < WorkerCount; i++)
        {
            Task.Run(() => WorkerLoop());
        }
    }

    public void EnqueueJob(VoxelJob job)
    {
        _workQueue.Enqueue(job);
    }

    public bool TryDequeueResult(out (VoxelJob job, object data) result)
    {
        return _resultsQueue.TryDequeue(out result);
    }

    private void WorkerLoop()
    {
        while (true)
        {
            if (_workQueue.TryDequeue(out VoxelJob job))
            {
                try
                {
                    if (_world.VoxelChunks.TryGetValue(job.Position, out Chunk chunk))
                    {
                        object resultData = null; // Can hold MeshData or be null

                        switch (job.JobType)
                        {
                            case VoxelJobType.GenerateData:
                                if (!chunk.IsDataGenerated)
                                {
                                    chunk.PrepareData(_terrainNoise, _caveNoise);
                                }
                                break;

                            case VoxelJobType.GenerateMesh:
                                resultData = chunk.GenerateMesh();
                                break;
                        }

                        _resultsQueue.Enqueue((job, resultData));
                    }
                }
                catch (System.Exception e)
                {
                    GD.PrintErr($"Worker thread crashed on chunk {job.Position}: {e.Message}\n{e.StackTrace}");
                }
            }
            else
            {
                System.Threading.Thread.Sleep(10);
            }
        }
    }
}
