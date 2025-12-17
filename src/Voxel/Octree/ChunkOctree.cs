using Godot;
using System.Collections.Generic;
using System.Linq;

public class ChunkOctree
{
    private readonly Dictionary<Vector3I, OctreeNode> _roots = new Dictionary<Vector3I, OctreeNode>();
    private readonly World _world;
    private readonly ChunkPool _pool;
    
    // Root covers a massive area to serve as the top-level container.
    // LOD 6 = 32 * 64 = 2048 units per chunk.
    // If we use LOD 7, it's 4096. LOD 8 is 8192.
    // Let's use LOD 7 (4096 units per root node) to reduce the number of roots needed.
    private const int RootLod = 7; 
    
    public ChunkOctree(World world, ChunkPool pool)
    {
        _world = world;
        _pool = pool;
    }

    public void Update(Vector3 logicalPlayerPos)
    {
        // Calculate the Root Coordinate (at RootLod) that contains the player
        float rootSize = (1 << RootLod) * _world.ActiveConfig.ChunkSize * _world.ActiveConfig.VoxelScale;
        
        Vector3I centerRootCoord = new Vector3I(
            Mathf.FloorToInt(logicalPlayerPos.X / rootSize),
            Mathf.FloorToInt(logicalPlayerPos.Y / rootSize),
            Mathf.FloorToInt(logicalPlayerPos.Z / rootSize)
        );
        
        // Ensure roots exist in a 2x2x2 (or 3x3x3) area around the player to prevent popping at boundaries
        int radius = 1; // 1 means check center + neighbors (-1 to 1) -> 3x3x3 grid of Huge Roots
        
        // 1. Create/Update Roots
        for (int x = -radius; x <= radius; x++)
        {
            for (int y = -radius; y <= radius; y++)
            {
                for (int z = -radius; z <= radius; z++)
                {
                    Vector3I rootPos = centerRootCoord + new Vector3I(x, y, z);
                    
                    if (!_roots.TryGetValue(rootPos, out OctreeNode root))
                    {
                        root = new OctreeNode(rootPos, RootLod);
                        _roots.Add(rootPos, root);
                    }
                    
                    // Update this root
                    // Note: We pass logicalPlayerPos, and OctreeNode uses this relative to its position.
                    // We must ensure OctreeNode handles the VoxelScale correctly in its distance checks.
                    root.Update(logicalPlayerPos, _world, _pool, isVisible: true);
                }
            }
        }

        // 2. Prune Roots (Garbage Collect far away roots)
        // We iterate a copy to modify dictionary
        var keys = _roots.Keys.ToList();
        foreach (var key in keys)
        {
            // Simple distance check in Root Grid Space
            int dx = Mathf.Abs(key.X - centerRootCoord.X);
            int dy = Mathf.Abs(key.Y - centerRootCoord.Y);
            int dz = Mathf.Abs(key.Z - centerRootCoord.Z);
            
            if (dx > radius + 1 || dy > radius + 1 || dz > radius + 1)
            {
                _roots[key].DisposeRecursive(_world, _pool);
                _roots.Remove(key);
            }
        }
    }
}

public class OctreeNode
{
    public OctreeNode[] Children;
    public Chunk ChunkData;
    public readonly Vector3I Position; // Position in Chunk Coordinates at this LOD
    public readonly int Lod;
    public bool IsSplit;
    
    public OctreeNode(Vector3I position, int lod)
    {
        Position = position;
        Lod = lod;
    }

    public void Update(Vector3 logicalPlayerPos, World world, ChunkPool pool, bool isVisible)
    {
        // Size in World Units (Meters)
        float size = (1 << Lod) * world.ActiveConfig.ChunkSize * world.ActiveConfig.VoxelScale;
        
        // Center in Logical World Units
        // Position * Size gives the bottom-left corner of the node in World Units
        Vector3 min = (Vector3)Position * size;
        Vector3 max = min + (Vector3.One * size);
        
        // Calculate Distance to AABB
        float dx = Mathf.Max(0, Mathf.Max(min.X - logicalPlayerPos.X, logicalPlayerPos.X - max.X));
        float dy = Mathf.Max(0, Mathf.Max(min.Y - logicalPlayerPos.Y, logicalPlayerPos.Y - max.Y));
        float dz = Mathf.Max(0, Mathf.Max(min.Z - logicalPlayerPos.Z, logicalPlayerPos.Z - max.Z));
        
        float dist = Mathf.Sqrt(dx*dx + dy*dy + dz*dz);

        // Split criteria:
        // 1. Standard structural split (1.5x size) to maintain hierarchy.
        // 2. RenderDistance enforcement: Ensure we split if we are within the user's requested LOD 0 radius.
        
        float baseChunkSize = world.ActiveConfig.ChunkSize * world.ActiveConfig.VoxelScale;
        float requestedLod0Radius = world.ActiveConfig.RenderDistance * baseChunkSize;
        
        // We must split if we are closer than the standard ratio OR if we are inside the requested LOD 0 zone.
        float splitDist = Mathf.Max(size * 1.5f, requestedLod0Radius);
        
        // Merge distance slightly larger for hysteresis
        float mergeDist = splitDist + (size * 0.5f); 

        bool shouldSplit = (Lod > 0) && (dist < splitDist);
        bool shouldMerge = (dist > mergeDist);

        if (IsSplit)
        {
             if (shouldMerge)
             {
                 // Merge Transition: We want to go to Leaf State.
                 // But we can only switch if Parent (Me) is ready.
                 
                 EnsureChunkLoaded(world, pool);
                 
                 if (ChunkData != null && ChunkData.State == ChunkState.Ready)
                 {
                     // Parent Ready -> Execute Merge
                     Merge(world, pool);
                     IsSplit = false;
                     
                     // Now I am a Leaf. I am visible if my parent allows it.
                     ChunkData.SetVisible(isVisible);
                 }
                 else
                 {
                     // Parent Loading -> Wait.
                     // Children must stay visible.
                     if (ChunkData != null) ChunkData.SetVisible(false);
                     
                     for (int i = 0; i < 8; i++)
                     {
                         Children[i].Update(logicalPlayerPos, world, pool, isVisible);
                     }
                 }
             }
             else
             {
                 // Maintain Split State (Inner Node)
                 // I am hidden. My children handle visibility.
                 
                 if (ChunkData != null) ChunkData.SetVisible(false);
                 
                 for (int i = 0; i < 8; i++)
                 {
                     Children[i].Update(logicalPlayerPos, world, pool, isVisible);
                 }
             }
        }
        else // Not Split (Leaf State)
        {
            if (shouldSplit)
            {
                Split(world, pool);
                
                // Split Transition: We want to go to Split State.
                // But we can only switch if ALL Children are ready.
                
                bool allChildrenReady = true;
                for (int i = 0; i < 8; i++)
                {
                    // Update children (load them)
                    Children[i].Update(logicalPlayerPos, world, pool, isVisible: false);
                    if (!Children[i].IsReadyOrSplit()) allChildrenReady = false;
                }

                if (allChildrenReady)
                {
                    // Children Ready -> Execute Split
                    IsSplit = true;
                    if (ChunkData != null) ChunkData.SetVisible(false);
                    
                    foreach (var child in Children) child.SetVisibilityRecursive(isVisible);
                }
                else
                {
                    // Children Loading -> Wait.
                    // I (Parent) must stay visible.
                    EnsureChunkLoaded(world, pool);
                    if (ChunkData != null && ChunkData.State == ChunkState.Ready)
                    {
                        ChunkData.SetVisible(isVisible);
                    }
                }
            }
            else
            {
                // Stable Leaf State
                EnsureChunkLoaded(world, pool);
                if (ChunkData != null && ChunkData.State == ChunkState.Ready)
                {
                    ChunkData.SetVisible(isVisible);
                }
            }
        }
    }

    private void Split(World world, ChunkPool pool)
    {
        if (Children != null) return;

        Children = new OctreeNode[8];
        int childLod = Lod - 1;
        
        for (int i = 0; i < 8; i++)
        {
            int x = (i & 1) == 1 ? 1 : 0;
            int y = (i & 2) == 2 ? 1 : 0;
            int z = (i & 4) == 4 ? 1 : 0;
            
            var childPos = new Vector3I(
                Position.X * 2 + x,
                Position.Y * 2 + y,
                Position.Z * 2 + z
            );
            
            Children[i] = new OctreeNode(childPos, childLod);
        }
    }

    private void Merge(World world, ChunkPool pool)
    {
        if (Children == null) return;

        for (int i = 0; i < 8; i++)
        {
            Children[i].DisposeRecursive(world, pool);
        }
        Children = null;
    }

    private void EnsureChunkLoaded(World world, ChunkPool pool)
    {
        if (ChunkData != null) return;

        ChunkData = pool.Get(Position);
        ChunkData.Lod = Lod;
        
        var key = new Vector4I(Position.X, Position.Y, Position.Z, Lod);
        world.VoxelChunks.TryAdd(key, ChunkData);
        
        // Pass VoxelScale correctly
        ChunkData.Initialize(world.GetWorld3D().Scenario, world.GetWorld3D().Space, world.VoxelMaterial, world.ActiveConfig.VoxelScale);
        
        // Position Mesh using Visual Coordinates (Logical - Offset)
        // Position is Logical Index.
        // Size of this LOD chunk = ChunkSize * VoxelScale * (1<<LOD)
        
        float lodScale = 1 << Lod;
        // The Chunk.Initialize uses (1<<Lod) * globalScale internal scaling for the mesh itself.
        // So we just need to place it at the right world position.
        
        Vector3 logicalPos = (Vector3)Position * world.ActiveConfig.ChunkSize * world.ActiveConfig.VoxelScale * lodScale;
        
        // Floating Origin Fix:
        // We need to access _worldOriginOffset. 
        // Since World exposes it? No, it's private.
        // BUT, we can calculate it: Logical - Visual = Offset.
        // Or refactor World to expose it.
        // Or assume the chunk will be positioned by World? 
        // No, Octree is doing it now.
        
        // Let's expose World.WorldOriginOffset public read-only.
        Vector3 visualPos = logicalPos - world.WorldOriginOffset;
        
        ChunkData.MeshInstance.Position = visualPos;
        
        world.RequestChunkLoad(ChunkData);
    }

    public void DisposeRecursive(World world, ChunkPool pool)
    {
        if (Children != null)
        {
            for (int i = 0; i < 8; i++)
            {
                Children[i].DisposeRecursive(world, pool);
            }
            Children = null;
        }

        if (ChunkData != null)
        {
            var key = new Vector4I(Position.X, Position.Y, Position.Z, Lod);
            world.VoxelChunks.TryRemove(key, out _);
            
            ChunkData.Unload();
            pool.Return(ChunkData);
            ChunkData = null;
        }
    }

    public bool IsReadyOrSplit()
    {
        if (IsSplit) return true; 
        return ChunkData != null && ChunkData.State == ChunkState.Ready;
    }

    public void SetVisibilityRecursive(bool visible)
    {
        if (IsSplit)
        {
            if (ChunkData != null) ChunkData.SetVisible(false);
            if (Children != null)
            {
                foreach (var child in Children) child.SetVisibilityRecursive(visible);
            }
        }
        else
        {
             if (ChunkData != null && ChunkData.State == ChunkState.Ready)
             {
                 ChunkData.SetVisible(visible);
             }
        }
    }
}
