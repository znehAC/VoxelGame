using System.Threading;

// Any class that can generate a mesh from voxel data must implement this interface.
public interface BaseMesher
{
    MeshData GenerateMeshData(MeshJobData jobData, VoxelDefinition[] definitions, CancellationToken token);
}
