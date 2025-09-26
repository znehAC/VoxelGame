using Godot;
using System.Collections.Concurrent;
using System.Threading.Tasks;

public class VoxelMesherPool
{
    public int WorkerCount { get; private set; }
    public int WorkQueueCount => _workQueue.Count;

    private readonly World _world;
    private readonly ConcurrentQueue<Vector3I> _workQueue = new();
    private readonly ConcurrentQueue<(Vector3I, MeshData)> _resultsQueue = new();

    private readonly FastNoiseLite _terrainNoise;
    private readonly FastNoiseLite _caveNoise;

    public VoxelMesherPool(World world, FastNoiseLite terrainNoise, FastNoiseLite caveNoise)
    {
        _world = world;
        _terrainNoise = terrainNoise;
        _caveNoise = caveNoise;
    }

    public void Start()
    {
        WorkerCount = System.Math.Max(1, System.Environment.ProcessorCount);

        GD.Print($"Starting {WorkerCount} meshing workers.");

        for (int i = 0; i < WorkerCount; i++)
        {
            Task.Run(() => WorkerLoop());
        }
    }

    public void EnqueueJob(Vector3I chunkPosition)
    {
        _workQueue.Enqueue(chunkPosition);
    }

    public bool TryDequeueResult(out (Vector3I, MeshData) result)
    {
        return _resultsQueue.TryDequeue(out result);
    }

    private void WorkerLoop()
    {
        while (true)
        {
            if (_workQueue.TryDequeue(out Vector3I chunkPosition))
            {
                try
                {
                    if (_world.VoxelChunks.TryGetValue(chunkPosition, out Chunk chunk))
                    {
                        if (!chunk.IsDataGenerated)
                        {
                            chunk.PrepareData(_terrainNoise, _caveNoise);
                        }

                        MeshData meshData = chunk.GenerateMesh();
                        _resultsQueue.Enqueue((chunkPosition, meshData));
                    }
                }
                catch (System.Exception e)
                {
                    GD.PrintErr($"Worker thread crashed on chunk {chunkPosition}: {e.Message}\n{e.StackTrace}");
                }
                System.Threading.Thread.Sleep(100);
            }
            else
            {
                System.Threading.Thread.Sleep(10);
            }
        }
    }
}
