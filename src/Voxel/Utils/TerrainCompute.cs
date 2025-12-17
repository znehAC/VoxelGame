using Godot;
using System;
using System.IO;
using System.Linq;

public partial class TerrainCompute : RefCounted
{
    private RenderingDevice _rd;
    private Rid _shader;
    private Rid _pipeline;
    private Rid _voxelBuffer;
    private Rid _metadataBuffer;
    private Rid _paramsBuffer; // Binding 2
    private Rid _uniformSet;
    private object _lock = new object();
    private volatile bool _initialized = false;
    private const int BufferSize = Chunk.Size * Chunk.Size * Chunk.Size * 4; // Cached buffer size

    private static bool _debugPrinted = false;

    public void Initialize(string shaderSource)
    {
        lock (_lock)
        {
            if (_initialized) return;
            GD.Print($"TerrainCompute: Initializing on Thread {System.Environment.CurrentManagedThreadId}...");
            
            _rd = RenderingServer.CreateLocalRenderingDevice();
            if (_rd == null)
            {
                GD.PrintErr("TerrainCompute: Failed to create Local RenderingDevice!");
                return;
            }

            shaderSource = shaderSource.Replace("#[compute]", "").Trim();

            var shaderSourceStruct = new RDShaderSource();
            shaderSourceStruct.SourceCompute = shaderSource;

            var shaderSpirv = _rd.ShaderCompileSpirVFromSource(shaderSourceStruct);
            
            _shader = _rd.ShaderCreateFromSpirV(shaderSpirv);
            if (!_shader.IsValid)
            {
                GD.PrintErr("TerrainCompute: Failed to create shader from SPIR-V.");
                return;
            }

            _pipeline = _rd.ComputePipelineCreate(_shader);
            if (!_pipeline.IsValid)
            {
                GD.PrintErr("TerrainCompute: Failed to create compute pipeline.");
                return;
            }

            // --- Allocate Resources Once ---
            
            // 1. Voxel Buffer (Binding 0)
            var initialData = new byte[BufferSize]; 
            _voxelBuffer = _rd.StorageBufferCreate((uint)BufferSize, initialData);

            var uniformVoxels = new RDUniform
            {
                UniformType = RenderingDevice.UniformType.StorageBuffer,
                Binding = 0
            };
            uniformVoxels.AddId(_voxelBuffer);

            // 2. Metadata Buffer (Binding 1)
            var initialMetadata = new byte[4]; // 1 uint
            _metadataBuffer = _rd.StorageBufferCreate(4, initialMetadata);
            
            var uniformMetadata = new RDUniform
            {
                UniformType = RenderingDevice.UniformType.StorageBuffer,
                Binding = 1
            };
            uniformMetadata.AddId(_metadataBuffer);

            // 3. Params Buffer (Binding 2)
            var initialParams = new byte[36]; // 9 ints * 4 bytes
            _paramsBuffer = _rd.StorageBufferCreate(36, initialParams);

            var uniformParams = new RDUniform
            {
                UniformType = RenderingDevice.UniformType.StorageBuffer,
                Binding = 2
            };
            uniformParams.AddId(_paramsBuffer);

            // Create Uniform Set with all 3 buffers
            _uniformSet = _rd.UniformSetCreate(new Godot.Collections.Array<RDUniform> { uniformVoxels, uniformMetadata, uniformParams }, _shader, 0);

            _initialized = true;
            GD.Print("TerrainCompute: Initialized successfully with cached resources.");
        }
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            if (_rd != null)
            {
                if (_uniformSet.IsValid) _rd.FreeRid(_uniformSet);
                if (_voxelBuffer.IsValid) _rd.FreeRid(_voxelBuffer);
                if (_metadataBuffer.IsValid) _rd.FreeRid(_metadataBuffer);
                if (_paramsBuffer.IsValid) _rd.FreeRid(_paramsBuffer);
                if (_pipeline.IsValid) _rd.FreeRid(_pipeline);
                
                _rd.Free();
                _rd = null;
            }
        }
        base.Dispose(disposing);
    }

    public (byte[] voxels, bool isEmpty) Generate(Vector3I chunkPos, int seed, int scale, int lod)
    {
        if (!_initialized) 
        {
            GD.PrintErr("TerrainCompute: Not initialized! Call Initialize(source) first.");
            return (new byte[Chunk.Size * Chunk.Size * Chunk.Size], true);
        }

        // 1. Prepare Parameters
        var paramData = new int[] { 
            chunkPos.X, chunkPos.Y, chunkPos.Z,
            Chunk.Size,
            seed,
            50,
            100,
            scale,
            lod
        };
        
        var paramBytes = new byte[paramData.Length * 4];
        Buffer.BlockCopy(paramData, 0, paramBytes, 0, paramBytes.Length);

        // Update Params Buffer
        _rd.BufferUpdate(_paramsBuffer, 0, (uint)paramBytes.Length, paramBytes);

        // Reset Metadata Buffer to 0
        _rd.BufferUpdate(_metadataBuffer, 0, 4, new byte[4]);

        // 2. Dispatch
        var computeList = _rd.ComputeListBegin();
        _rd.ComputeListBindComputePipeline(computeList, _pipeline);
        _rd.ComputeListBindUniformSet(computeList, _uniformSet, 0);
        // NO Push Constant anymore
        
        uint groups = (uint)Chunk.Size / 4;
        _rd.ComputeListDispatch(computeList, groups, groups, groups);
        
        _rd.ComputeListEnd();

        // 3. Sync and Read
        _rd.Submit();
        _rd.Sync(); 
        
        byte[] outputBytes = _rd.BufferGetData(_voxelBuffer);
        byte[] metadataBytes = _rd.BufferGetData(_metadataBuffer);

        if (!_debugPrinted && outputBytes != null && outputBytes.Length > 0)
        {
            _debugPrinted = true;
            // Print first 24 bytes (6 ints: Size, cx, cy, cz, scale, lod)
            GD.Print($"[GPU DEBUG] Chunk {chunkPos} LOD {lod} Scale {scale} | NonEmptyFlag: {BitConverter.ToInt32(metadataBytes, 0)} | Bytes: {string.Join("-", outputBytes.Take(24))}");
        }
        
        if (outputBytes == null || outputBytes.Length != BufferSize)
        {
            GD.PrintErr($"TerrainCompute: Buffer size mismatch! Expected {BufferSize}, got {outputBytes?.Length ?? -1}.");
            return (new byte[Chunk.Size * Chunk.Size * Chunk.Size], true);
        }

        int nonAirFlag = BitConverter.ToInt32(metadataBytes, 0);
        bool isEmpty = (nonAirFlag == 0);

        int voxelCount = Chunk.Size * Chunk.Size * Chunk.Size;
        byte[] voxels = new byte[voxelCount];
        for (int i = 0; i < voxelCount; i++)
        {
            voxels[i] = outputBytes[i * 4]; 
        }

        return (voxels, isEmpty);
    }
}
