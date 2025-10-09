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

    public static readonly VoxelDefinition[] Definitions = new VoxelDefinition[]
    {
        // 0: Air
        new VoxelDefinition("Air", new Color(0,0,0,0), isSolid: false),
        
        // 1: Dirt
        new VoxelDefinition("Dirt", new Color(0.54f, 0.27f, 0.07f), isSolid: true),
        
        // 2: Stone
        new VoxelDefinition("Stone", new Color(0.5f, 0.5f, 0.5f), isSolid: true),

        // 3: Grass
        new VoxelDefinition("Dirt", new Color(0.54f, 0.27f, 0.07f), isSolid: true),

        // 4: Water
        new VoxelDefinition("Dirt", new Color(0.54f, 0.27f, 0.07f), isSolid: true),
    };
}
