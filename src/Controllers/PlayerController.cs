using Godot;

public partial class PlayerController : CharacterBody3D
{
    [Export]
    public float MouseSensitivity { get; set; } = 0.5f;
    [Export]
    public float MoveSpeed { get; set; } = 200.0f; // Adjusted for 10cm voxels
    [Export]
    public float JumpVelocity { get; set; } = 15.0f;

    // Gravity is based on Godot's project settings by default.
    public float Gravity = ProjectSettings.GetSetting("physics/3d/default_gravity").AsSingle();

    // We need a reference to the camera to rotate it for looking up/down.
    private Camera3D _camera;
    private Vector2 _mouseDelta;

    public override void _Ready()
    {
        _camera = GetNode<Camera3D>("Camera3D"); // Get the child camera node
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
    }

    public override void _PhysicsProcess(double delta)
    {
        // --- Mouselook Rotation ---
        // We handle this first. Note the separation of concerns.
        if (Input.MouseMode == Input.MouseModeEnum.Captured)
        {
            // Rotate the entire CharacterBody3D for left/right (yaw)
            RotateY(Mathf.DegToRad(-_mouseDelta.X * MouseSensitivity));

            // Only rotate the Camera3D for up/down (pitch)
            _camera.RotateX(Mathf.DegToRad(-_mouseDelta.Y * MouseSensitivity));

            // Clamp the camera's rotation to prevent it from flipping upside down
            var cameraRotation = _camera.RotationDegrees;
            cameraRotation.X = Mathf.Clamp(cameraRotation.X, -90.0f, 90.0f);
            _camera.RotationDegrees = cameraRotation;
        }
        _mouseDelta = Vector2.Zero; // Reset delta each frame

        // --- Movement and Physics ---
        Vector3 velocity = Velocity;

        // Apply gravity.
        if (!IsOnFloor())
        {
            velocity.Y -= Gravity * (float)delta;
        }

        // Handle Jump.
        if (Input.IsActionJustPressed("jump") && IsOnFloor())
        {
            velocity.Y = JumpVelocity;
        }

        // Get the input direction and handle the movement/deceleration.
        Vector2 inputDir = Input.GetVector("move_left", "move_right", "move_forward", "move_backward");
        // Use the CharacterBody's basis to move in the direction it's facing.
        Vector3 direction = (Transform.Basis * new Vector3(inputDir.X, 0, inputDir.Y)).Normalized();

        if (direction != Vector3.Zero)
        {
            velocity.X = direction.X * MoveSpeed;
            velocity.Z = direction.Z * MoveSpeed;
        }
        else
        {
            // Simple friction/deceleration
            velocity.X = Mathf.MoveToward(Velocity.X, 0, MoveSpeed);
            velocity.Z = Mathf.MoveToward(Velocity.Z, 0, MoveSpeed);
        }

        // This is the magic line. We set the Velocity property, and then...
        Velocity = velocity;
        // ...Godot's physics engine does the rest.
        MoveAndSlide();
    }
}
