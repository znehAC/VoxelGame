use crate::gpu::GpuContext;
use ara_core::bytemuck;

/// Why the light map needs recomputation.
#[derive(Debug, Clone)]
pub enum LightDirtyReason {
    /// Initial load or chunk streaming
    ChunkLoaded,
    /// Block placed or removed
    BlockEdited { pos: ara_core::glam::IVec3 },
    /// Sun angle or sky color changed
    TimeOfDay,
}

pub struct LightPropagator {
    pipeline: wgpu::ComputePipeline,
    compaction_pipeline: wgpu::ComputePipeline,
    bind_group_a: wgpu::BindGroup,
    bind_group_b: wgpu::BindGroup,
    compaction_bind_group: wgpu::BindGroup,
    pub dirty_chunks_buf: wgpu::Buffer,
    pub origin_buf: wgpu::Buffer,
    pub light_grid_a: wgpu::Buffer,
    pub active_chunks_buf: wgpu::Buffer,
    pub indirect_args_buf: wgpu::Buffer,
    pub first_frame: bool,
    dirty: bool,
    propagation_iterations: u32,
}

impl LightPropagator {
    pub fn new(
        gpu: &GpuContext,
        top_grid_buf: &wgpu::Buffer,
        brick_pool_buf: &wgpu::Buffer,
        palette_view: &wgpu::TextureView,
        uniform_buf: &wgpu::Buffer,
        light_buf: &wgpu::Buffer,
    ) -> Self {
        let device = gpu.device();

        // 512^3 * 4 bytes = 536.8 MB per buffer
        let grid_vol = 512u64 * 512 * 512 * 4;
        let light_grid_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Toroidal Light Grid A"),
            size: grid_vol,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let light_grid_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Toroidal Light Grid B"),
            size: grid_vol,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 64^3 bits = 32,768 bytes
        let dirty_chunks_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dirty Chunk Bitmask"),
            size: 32_768,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initialize all chunks to dirty for the first frame
        let initial_dirty = vec![0xFFFFFFFFu32; 32_768 / 4];
        gpu.queue()
            .write_buffer(&dirty_chunks_buf, 0, bytemuck::cast_slice(&initial_dirty));

        let origin_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Toroidal Origin Uniform"),
            size: 16, // vec4<i32>
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initial origin at 0,0,0
        gpu.queue()
            .write_buffer(&origin_buf, 0, bytemuck::cast_slice(&[0i32, 0, 0, 0]));

        let active_chunks_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Active Chunks Buffer"),
            size: 262_144 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // indirect args: x (atomic counter / workgroup_x), y, z, pad
        let indirect_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Args Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Toroidal Propagate BGL"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let create_bg = |in_buf: &wgpu::Buffer, out_buf: &wgpu::Buffer, label: &str| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: top_grid_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: brick_pool_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: in_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: dirty_chunks_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: origin_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: active_chunks_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(palette_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: light_buf.as_entire_binding(),
                    },
                ],
            })
        };

        let bind_group_a = create_bg(&light_grid_a, &light_grid_b, "Toroidal BG A (A->B)");
        let bind_group_b = create_bg(&light_grid_b, &light_grid_a, "Toroidal BG B (B->A)");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Toroidal Propagate Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../assets/shaders/light_propagate.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Toroidal Propagate Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Toroidal Propagate Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("propagate"),
            compilation_options: Default::default(),
            cache: None,
        });

        let compaction_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compaction BGL"),
            entries: &[
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

        let compaction_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compaction BG"),
            layout: &compaction_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dirty_chunks_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: active_chunks_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: indirect_args_buf.as_entire_binding(),
                },
            ],
        });

        let compaction_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compaction Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../assets/shaders/compact_chunks.wgsl").into(),
            ),
        });

        let compaction_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compaction Layout"),
                bind_group_layouts: &[&compaction_bgl],
                push_constant_ranges: &[],
            });

        let compaction_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compaction Pipeline"),
                layout: Some(&compaction_pipeline_layout),
                module: &compaction_shader,
                entry_point: Some("compact"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            pipeline,
            compaction_pipeline,
            bind_group_a,
            bind_group_b,
            compaction_bind_group,
            dirty_chunks_buf,
            origin_buf,
            light_grid_a,
            active_chunks_buf,
            indirect_args_buf,
            first_frame: true,
            dirty: true,
            propagation_iterations: 20,
        }
    }

    /// Mark the light map as needing recomputation.
    pub fn mark_dirty(&mut self, _reason: LightDirtyReason) {
        self.dirty = true;
    }

    pub fn needs_update(&self) -> bool {
        self.dirty
    }

    pub fn set_iterations(&mut self, iterations: u32) {
        self.propagation_iterations = iterations;
    }

    /// Only execute propagation if marked dirty. Returns whether it ran.
    pub fn execute_if_dirty(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        gpu: &GpuContext,
    ) -> bool {
        if !self.dirty {
            return false;
        }
        let iters = self.propagation_iterations;
        self.execute(encoder, iters, gpu);
        self.dirty = false;
        true
    }

    pub fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        iterations: u32,
        gpu: &GpuContext,
    ) {
        // Reset the indirect dispatch args. x = atomic counter, y = 1, z = 1, pad = 0
        gpu.queue().write_buffer(
            &self.indirect_args_buf,
            0,
            bytemuck::cast_slice(&[0u32, 1, 1, 0]),
        );

        {
            let mut comp_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Toroidal Compaction Pass"),
                timestamp_writes: None,
            });
            comp_pass.set_pipeline(&self.compaction_pipeline);
            comp_pass.set_bind_group(0, &self.compaction_bind_group, &[]);
            comp_pass.dispatch_workgroups(1024, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Toroidal Propagation Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);

            for i in 0..iterations {
                if i % 2 == 0 {
                    pass.set_bind_group(0, &self.bind_group_a, &[]);
                } else {
                    pass.set_bind_group(0, &self.bind_group_b, &[]);
                }
                pass.dispatch_workgroups_indirect(&self.indirect_args_buf, 0);
            }
        }

        if self.first_frame {
            self.first_frame = false;
        } else {
            encoder.clear_buffer(&self.dirty_chunks_buf, 0, None);
        }
    }

    /// Mark a specific 8x8x8 chunk as dirty when a block is placed/broken
    pub fn mark_dirty_radius(
        &mut self,
        gpu: &GpuContext,
        world_pos: ara_core::glam::IVec3,
        radius: i32,
    ) {
        self.dirty = true;
        let origin = ara_core::glam::IVec3::ZERO; // Replace with actual tracked origin later
        let rel = world_pos - origin;

        let cx_base = rel.x.div_euclid(8);
        let cy_base = rel.y.div_euclid(8);
        let cz_base = rel.z.div_euclid(8);

        // A radius in standard chunk steps logic (e.g chunks up to 14 blocks / 8 = roughly radius 2)
        // Light travels ~14 voxels. The span of influenced chunks might be +/- 2 in each dimension.
        let ch_rad = (radius as f32 / 8.0).ceil() as i32;

        let mut local_map = std::collections::HashMap::new();

        for z in -ch_rad..=ch_rad {
            for y in -ch_rad..=ch_rad {
                for x in -ch_rad..=ch_rad {
                    let cx = (cx_base + x) & 63;
                    let cy = (cy_base + y) & 63;
                    let cz = (cz_base + z) & 63;

                    let chunk_idx = cz * 4096 + cy * 64 + cx;
                    let word_idx = (chunk_idx >> 5) as u32;
                    let bit_idx = (chunk_idx & 31) as u32;

                    let entry = local_map.entry(word_idx).or_insert(0u32);
                    *entry |= 1u32 << bit_idx;
                }
            }
        }

        // Extremely slow loop doing individual buffer writes just for proving the logic
        // But the user constraint implies batching isn't a strict requirement,
        // and we are CPU side so queue.write_buffer is asynchronous and cheap enough for a few bursts
        // when blocks are explicitly placed/broken by user.
        for (word_idx, mask) in local_map {
            let offset = (word_idx * 4) as wgpu::BufferAddress;
            gpu.queue().write_buffer(
                &self.dirty_chunks_buf,
                offset,
                bytemuck::cast_slice(&[mask]),
            );
        }
    }
}
