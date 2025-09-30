using Godot;

public partial class PlayerController : CharacterBody3D
{
    [Export]
    public float MouseSensitivity { get; set; } = 0.5f;
    [Export]
    public float MoveSpeed { get; set; } = 40.0f;
    [Export]
    public float JumpVelocity { get; set; } = 150.0f;
    [Export]
    public float FlySpeed { get; set; } = 100.0f;

    public float Gravity = ProjectSettings.GetSetting("physics/3d/default_gravity").AsSingle() * 30;

    private Camera3D _camera;
    private Vector2 _mouseDelta;
    private bool _isFlying = false;

    public override void _Ready()
    {
        _camera = GetNode<Camera3D>("Camera3D");
        Input.MouseMode = Input.MouseModeEnum.Captured;
    }

    public override void _Input(InputEvent e)
    {
        if (e is InputEventMouseMotion mouseMotion)
        {
            _mouseDelta = mouseMotion.Relative;
        }

        if (e.IsActionPressed("ui_cancel"))
        {
            Input.MouseMode = Input.MouseModeEnum.Visible;
        }
        if (e is InputEventMouseButton mouseButton && mouseButton.IsPressed() && mouseButton.ButtonIndex == MouseButton.Left)
        {
            Input.MouseMode = Input.MouseModeEnum.Captured;
        }
        if (e.IsActionPressed("toggle_fly"))
        {
            _isFlying = !_isFlying;
            GD.Print($"Flying: {_isFlying}");
        }
    }

    public override void _PhysicsProcess(double delta)
    {
        // --- Mouselook Rotation ---
        HandleMouseLook();
        _mouseDelta = Vector2.Zero; // Reset delta each frame

        // --- Movement and Physics ---
        Vector3 velocity = Velocity;
        Vector2 inputDir = Input.GetVector("move_left", "move_right", "move_forward", "move_backward");
        Vector3 direction = (Transform.Basis * new Vector3(inputDir.X, 0, inputDir.Y)).Normalized();

        if (_isFlying)
        {
            velocity = HandleFlying(direction, velocity);
        }
        else
        {
            velocity = HandleWalking(direction, velocity, (float)delta);
        }

        Velocity = velocity;
        MoveAndSlide();
    }

    private void HandleMouseLook()
    {
        if (Input.MouseMode != Input.MouseModeEnum.Captured) return;

        RotateY(Mathf.DegToRad(-_mouseDelta.X * MouseSensitivity));
        _camera.RotateX(Mathf.DegToRad(-_mouseDelta.Y * MouseSensitivity));

        var cameraRotation = _camera.RotationDegrees;
        cameraRotation.X = Mathf.Clamp(cameraRotation.X, -90.0f, 90.0f);
        _camera.RotationDegrees = cameraRotation;
    }

    private Vector3 HandleWalking(Vector3 direction, Vector3 velocity, float delta)
    {
        if (!IsOnFloor())
        {
            velocity.Y -= Gravity * delta;
        }

        if (Input.IsActionJustPressed("jump") && IsOnFloor())
        {
            velocity.Y = JumpVelocity;
        }

        if (direction != Vector3.Zero)
        {
            velocity.X = direction.X * MoveSpeed;
            velocity.Z = direction.Z * MoveSpeed;
        }
        else
        {
            velocity.X = Mathf.MoveToward(Velocity.X, 0, MoveSpeed);
            velocity.Z = Mathf.MoveToward(Velocity.Z, 0, MoveSpeed);
        }
        return velocity;
    }

    private Vector3 HandleFlying(Vector3 direction, Vector3 velocity)
    {
        // Horizontal movement
        if (direction != Vector3.Zero)
        {
            velocity.X = direction.X * FlySpeed;
            velocity.Z = direction.Z * FlySpeed;
        }
        else
        {
            velocity.X = Mathf.MoveToward(Velocity.X, 0, FlySpeed);
            velocity.Z = Mathf.MoveToward(Velocity.Z, 0, FlySpeed);
        }

        // Vertical movement
        if (Input.IsActionPressed("move_up"))
        {
            velocity.Y = FlySpeed;
        }
        else if (Input.IsActionPressed("move_down"))
        {
            velocity.Y = -FlySpeed;
        }
        else
        {
            velocity.Y = Mathf.MoveToward(Velocity.Y, 0, FlySpeed);
        }

        return velocity;
    }
}
