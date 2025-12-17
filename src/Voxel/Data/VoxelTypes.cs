using Godot;

public readonly struct VoxelDefinition
{
    public readonly string Name;
    public readonly Color VoxelColor;
    public readonly bool IsSolid;
    // public readonly Vector2I AtlasCoordTop;
    // public readonly Vector2I AtlasCoordSide;

    public VoxelDefinition(string name, Color color, bool isSolid)
    {
        Name = name;
        VoxelColor = color;
        IsSolid = isSolid;
    }
}

public static class VoxelTypes
{

    public const byte Air = 0;
    public const byte Dirt = 1;
    public const byte Stone = 2;
    public const byte Grass = 3;
    public const byte Water = 4;
    public const byte Snow = 5;
    public const byte Sand = 6;
    public const byte Wood = 7;
    public const byte RedStone = 8;
    public const byte BlueStone = 9;

    public static readonly VoxelDefinition[] Definitions = new VoxelDefinition[]
    {
        // 0: Air
        new VoxelDefinition("Air", new Color(0,0,0,0), isSolid: false),
        
        // 1: Dirt
        new VoxelDefinition("Dirt", new Color(0.54f, 0.27f, 0.07f), isSolid: true),
        
        // 2: Stone
        new VoxelDefinition("Stone", new Color(0.5f, 0.5f, 0.5f), isSolid: true),

        // 3: Grass
        new VoxelDefinition("Grass", new Color(0.0f, 1.0f, 0.0f), isSolid: true),

        // 4: Water
        new VoxelDefinition("Water", new Color(0.0f, 0.0f, 1.0f), isSolid: true),

        // 5: Snow (White)
        new VoxelDefinition("Snow", new Color(0.9f, 0.9f, 0.9f), isSolid: true),

        // 6: Sand (Yellow)
        new VoxelDefinition("Sand", new Color(0.94f, 0.90f, 0.55f), isSolid: true),

        // 7: Wood (Dark Brown)
        new VoxelDefinition("Wood", new Color(0.4f, 0.2f, 0.0f), isSolid: true),

        // 8: RedStone (Red)
        new VoxelDefinition("RedStone", new Color(0.8f, 0.1f, 0.1f), isSolid: true),

        // 9: BlueStone (Blue)
        new VoxelDefinition("BlueStone", new Color(0.1f, 0.1f, 0.8f), isSolid: true),
    };
}
