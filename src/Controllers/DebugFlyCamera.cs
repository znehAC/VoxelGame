using Godot;

public partial class DebugFlyCamera : Camera3D
{
    [Export]
    public float MouseSensitivity { get; set; } = 0.75f;

    [Export]
    public float MoveSpeed { get; set; } = 100.0f;

    [Export]
    public float SpeedStep { get; set; } = 10.0f;

    private Vector2 _mouseDelta;

    public override void _Ready()
    {
        Input.MouseMode = Input.MouseModeEnum.Captured;
    }

    public override void _Input(InputEvent e)
    {

        if (e is InputEventMouseMotion mouseMotion)
        {
            _mouseDelta = mouseMotion.Relative;
        }

        if (e is InputEventMouseButton mouseButtonEvent && mouseButtonEvent.IsPressed())
        {
            switch (mouseButtonEvent.ButtonIndex)
            {
                case MouseButton.WheelUp:
                    MoveSpeed += SpeedStep;
                    break;
                case MouseButton.WheelDown:
                    MoveSpeed = Mathf.Max(SpeedStep, MoveSpeed - SpeedStep);
                    break;
                case MouseButton.Left:
                    Input.MouseMode = Input.MouseModeEnum.Captured;
                    break;
            }
        }

        if (e.IsActionPressed("ui_cancel"))
        {
            Input.MouseMode = Input.MouseModeEnum.Visible;
        }
    }

    public override void _PhysicsProcess(double delta)
    {
        if (Input.MouseMode == Input.MouseModeEnum.Captured)
        {
            RotateY(Mathf.DegToRad(-_mouseDelta.X * MouseSensitivity));
            RotateObjectLocal(Vector3.Right, Mathf.DegToRad(-_mouseDelta.Y * MouseSensitivity));

            var newRotation = RotationDegrees;
            newRotation.X = Mathf.Clamp(newRotation.X, -90.0f, 90.0f);
            RotationDegrees = newRotation;
        }
        _mouseDelta = Vector2.Zero;

        var inputDir = Input.GetVector("move_left", "move_right", "move_forward", "move_backward");

        var direction = (Transform.Basis * new Vector3(inputDir.X, 0, inputDir.Y)).Normalized();

        if (Input.IsActionPressed("move_up"))
        {
            direction += Vector3.Up;
        }
        if (Input.IsActionPressed("move_down"))
        {
            direction += Vector3.Down;
        }

        Position += direction * MoveSpeed * (float)delta;
    }
}
