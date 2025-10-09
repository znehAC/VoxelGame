using Godot;
using System.Collections.Concurrent;

public class ChunkPool
{
    private readonly ConcurrentBag<Chunk> _pool = new();
    private readonly World _world;

    public ChunkPool(World world)
    {
        _world = world;
    }

    public Chunk Get(Vector3I position)
    {
        if (_pool.TryTake(out var chunk))
        {
            chunk.Reset(position);
            return chunk;
        }

        return new Chunk(_world, position);
    }

    public void Return(Chunk chunk)
    {
        _pool.Add(chunk);
    }
}
