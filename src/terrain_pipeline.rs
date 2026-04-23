//! GPU-driven terrain generation compute pipeline.

use crate::brick_map::BrickMap;
use crate::gpu::GpuContext;
use ara_core::bytemuck;

/// One terrain generation job — one brick to be filled by the GPU.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainGenJob {
    pub brick_idx: u32,
    pub brick_x: i32,
    pub brick_y: i32,
    pub brick_z: i32,
    pub lod: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Compute pipeline that fills brick_pool + brick_occupancy from a sine-wave height function.
pub struct TerrainPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    job_buf: wgpu::Buffer,
    max_jobs: u32,
}

impl TerrainPipeline {
    pub fn new(gpu: &GpuContext, _brick_map: &BrickMap, max_jobs: u32) -> Self {
        let device = gpu.device();

        let job_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain Gen Jobs"),
            size: max_jobs as u64 * std::mem::size_of::<TerrainGenJob>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Gen BGL"),
            entries: &[
                // 0: jobs (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: brick_pool (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: brick_occupancy (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Gen Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../assets/shaders/terrain_gen.wgsl").into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Terrain Gen Pipeline Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Terrain Gen Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("generate"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, bgl, job_buf, max_jobs }
    }

    pub fn max_jobs(&self) -> usize {
        self.max_jobs as usize
    }

    /// Dispatch terrain generation for the given jobs. Noop if empty.
    pub fn generate(
        &self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        jobs: &[TerrainGenJob],
        brick_map: &BrickMap,
    ) {
        if jobs.is_empty() {
            return;
        }
        debug_assert!(jobs.len() <= self.max_jobs as usize);

        gpu.queue()
            .write_buffer(&self.job_buf, 0, bytemuck::cast_slice(jobs));

        let bind_group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Gen Bind Group"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.job_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: brick_map.brick_pool_buf().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: brick_map.brick_occupancy_buf().as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Terrain Gen Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(jobs.len() as u32, 1, 1);
    }
}
