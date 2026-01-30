using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

/// <summary>
/// Manages grid-based placement of gate objects in the maze
/// Supports semantic parameter-based positioning with seed control
/// </summary>
public class GateGridManager : MonoBehaviour
{
    [Header("Grid Configuration")]
    [SerializeField] private int gridWidth = 20;
    [SerializeField] private int gridHeight = 20;
    [SerializeField] private float cellSize = 1.5f;
    
    [Header("Gate Configuration")]
    [SerializeField] private int numberOfGates = 25;
    [SerializeField] private string levelName = "default";
    [SerializeField] private string objFolderPath = "PythonScripts/ScientificComputing/Output";
    
    [Header("Semantic Parameters")]
    [Tooltip("Seed for deterministic random generation")]
    [SerializeField] private int seed = 42;
    
    [Tooltip("Influences gate rotation distribution (0-1, 0.5 = uniform)")]
    [SerializeField, Range(0f, 1f)] private float rotationBias = 0.5f;
    
    [Tooltip("Influences gate clustering (0-1, 0 = spread out, 1 = clustered)")]
    [SerializeField, Range(0f, 1f)] private float clusteringFactor = 0.3f;
    
    [Tooltip("Minimum distance between gate centers in grid cells")]
    [SerializeField] private int minGateDistance = 0;
    
    [Tooltip("Influences gate scale variation (0 = no variation, 1 = high variation)")]
    [SerializeField, Range(0f, 1f)] private float scaleVariation = 0.1f;
    
    [Header("Pathway Configuration")]
    [Tooltip("Ensures pathways by reserving cells (experimental)")]
    [SerializeField] private bool enforcePathways = false;
    
    [Tooltip("Percentage of cells reserved as pathways (0-1)")]
    [SerializeField, Range(0f, 0.5f)] private float pathwayReserve = 0.15f;
    
    [Header("Debug")]
    [SerializeField] private bool showGridGizmos = true;
    [SerializeField] private bool autoGenerateOnStart = true;
    
    [Header("Position Adjustment")]
    [Tooltip("Manual offset to adjust gate world positions if they don't align with grid")]
    [SerializeField] private Vector3 positionOffset = Vector3.zero;
    
    [Tooltip("Adjust if gate pivot is not at center (e.g., at bottom = 0.5 for Y)")]
    [SerializeField] private Vector3 pivotOffset = Vector3.zero;
    
    [Header("Physics")]
    [Tooltip("Physics material to apply to all gate colliders (e.g., no-friction)")]
    [SerializeField] private PhysicsMaterial gatePhysicsMaterial;
    
    [Tooltip("Make gate colliders triggers (non-solid)")]
    [SerializeField] private bool gateTriggers = false;
    
    // Internal data structures
    private bool[,] occupiedCells;
    private List<GatePlacement> placedGates = new List<GatePlacement>();
    private GameObject gateModel;
    private Vector3 gateBounds;
    private Vector3 gatePivotPoint;
    
    private class GatePlacement
    {
        public Vector3 position;
        public Quaternion rotation;
        public Vector2Int gridPosition;
        public int cellsOccupiedWidth;
        public int cellsOccupiedHeight;
        public GameObject gameObject;
    }
    
    void Start()
    {
        if (autoGenerateOnStart)
        {
            GenerateMaze();
        }
    }
    
    /// <summary>
    /// Main function to generate the maze with gates
    /// </summary>
    public void GenerateMaze()
    {
        ClearExistingGates();
        InitializeGrid();
        LoadGateModel();
        
        if (gateModel == null)
        {
            Debug.LogError($"Failed to load gate model for level '{levelName}'");
            return;
        }
        
        CalculateGateBounds();
        PlaceGates();
        
        Debug.Log($"Maze generated with {placedGates.Count} gates using seed {seed}");
    }
    
    /// <summary>
    /// Clear all previously placed gates
    /// </summary>
    private void ClearExistingGates()
    {
        foreach (var gate in placedGates)
        {
            if (gate.gameObject != null)
            {
                DestroyImmediate(gate.gameObject);
            }
        }
        placedGates.Clear();
    }
    
    /// <summary>
    /// Initialize the grid occupancy array
    /// </summary>
    private void InitializeGrid()
    {
        occupiedCells = new bool[gridWidth, gridHeight];
        
        // Reserve cells for pathways if enabled
        if (enforcePathways)
        {
            Random.InitState(seed);
            int cellsToReserve = Mathf.RoundToInt(gridWidth * gridHeight * pathwayReserve);
            
            for (int i = 0; i < cellsToReserve; i++)
            {
                int x = Random.Range(0, gridWidth);
                int y = Random.Range(0, gridHeight);
                occupiedCells[x, y] = true; // Mark as occupied to prevent gate placement
            }
        }
    }
    
    /// <summary>
    /// Load the gate model from the OBJ file
    /// </summary>
    private void LoadGateModel()
    {
        string modelPath = $"{objFolderPath}arch_{levelName}";
        
        // Try loading from Resources folder
        gateModel = Resources.Load<GameObject>(modelPath);
        
        if (gateModel == null)
        {
            Debug.LogWarning($"Could not load gate model from Resources at '{modelPath}'. " +
                           "Make sure the model is in a Resources folder or use a runtime OBJ importer.");
            
            // Create a placeholder cube for testing
            gateModel = GameObject.CreatePrimitive(PrimitiveType.Cube);
            gateModel.transform.localScale = new Vector3(2f, 3f, 0.5f);
            Debug.LogWarning("Using placeholder cube for gate model");
        }
    }
    
    /// <summary>
    /// Calculate how many grid cells the gate occupies based on its bounds
    /// </summary>
    private void CalculateGateBounds()
    {
        // Get bounds from mesh or collider
        Renderer renderer = gateModel.GetComponentInChildren<Renderer>();
        if (renderer != null)
        {
            Bounds bounds = renderer.bounds;
            gateBounds = bounds.size;
            
            // Calculate where the pivot is relative to the bounds
            // This helps position gates correctly on the grid
            gatePivotPoint = gateModel.transform.position - bounds.center;
        }
        else
        {
            // Default size if no renderer found
            gateBounds = new Vector3(2f, 3f, 0.5f);
            gatePivotPoint = Vector3.zero;
            Debug.LogWarning("No renderer found on gate model, using default bounds");
        }
        
        Debug.Log($"Gate bounds: {gateBounds}, Pivot offset: {gatePivotPoint}, Cell size: {cellSize}");
    }
    
    /// <summary>
    /// Calculate how many cells a gate occupies based on rotation
    /// </summary>
    private void GetGateCellOccupancy(int rotationIndex, out int widthCells, out int heightCells)
    {
        // rotationIndex: 0=0°, 1=90°, 2=180°, 3=270°
        bool isRotated = (rotationIndex % 2 == 1); // 90° or 270°
        
        if (isRotated)
        {
            // Width and height swap when rotated 90° or 270°
            widthCells = Mathf.CeilToInt(gateBounds.z / cellSize);
            heightCells = Mathf.CeilToInt(gateBounds.x / cellSize);
        }
        else
        {
            widthCells = Mathf.CeilToInt(gateBounds.x / cellSize);
            heightCells = Mathf.CeilToInt(gateBounds.z / cellSize);
        }
        
        // Ensure at least 1 cell
        widthCells = Mathf.Max(1, widthCells);
        heightCells = Mathf.Max(1, heightCells);
    }
    
    /// <summary>
    /// Place gates on the grid using semantic parameters
    /// </summary>
    private void PlaceGates()
    {
        Random.InitState(seed);
        int attemptedPlacements = 0;
        int maxAttempts = numberOfGates * 50; // Prevent infinite loops
        
        while (placedGates.Count < numberOfGates && attemptedPlacements < maxAttempts)
        {
            attemptedPlacements++;
            
            // Determine rotation (0, 90, 180, 270 degrees)
            int rotationIndex = GetSemanticRotation();
            int widthCells, heightCells;
            GetGateCellOccupancy(rotationIndex, out widthCells, out heightCells);
            
            // Find a valid position
            Vector2Int gridPos = FindValidGridPosition(widthCells, heightCells);
            
            if (gridPos.x == -1) // No valid position found
            {
                continue;
            }
            
            // Check distance constraint
            if (!CheckMinimumDistance(gridPos))
            {
                continue;
            }
            
            // Mark cells as occupied
            MarkCellsOccupied(gridPos, widthCells, heightCells, true);
            
            // Create the gate instance
            Vector3 worldPos = GridToWorldPosition(gridPos, widthCells, heightCells);
            Quaternion rotation = Quaternion.Euler(0, rotationIndex * 90f, 0);
            
            GameObject gateInstance = Instantiate(gateModel, worldPos, rotation, transform);
            gateInstance.name = $"Gate_{placedGates.Count}_{levelName}";
            
            // Apply physics material to colliders
            ApplyPhysicsMaterialToGate(gateInstance);
            
            // Apply scale variation
            if (scaleVariation > 0)
            {
                float scale = 1f + Random.Range(-scaleVariation, scaleVariation);
                gateInstance.transform.localScale *= scale;
            }
            
            // Store placement data
            GatePlacement placement = new GatePlacement
            {
                position = worldPos,
                rotation = rotation,
                gridPosition = gridPos,
                cellsOccupiedWidth = widthCells,
                cellsOccupiedHeight = heightCells,
                gameObject = gateInstance
            };
            
            placedGates.Add(placement);
        }
        
        if (placedGates.Count < numberOfGates)
        {
            Debug.LogWarning($"Could only place {placedGates.Count}/{numberOfGates} gates after {attemptedPlacements} attempts");
        }
    }
    
    /// <summary>
    /// Get rotation based on semantic parameters
    /// </summary>
    private int GetSemanticRotation()
    {
        // Use rotationBias to influence distribution
        // 0.5 = uniform distribution across all 4 rotations
        // < 0.5 = bias toward 0° and 180°
        // > 0.5 = bias toward 90° and 270°
        
        float rand = Random.value;
        
        if (rotationBias < 0.5f)
        {
            // Bias toward 0° and 180°
            float bias = 1f - (rotationBias * 2f); // 0 to 1
            if (rand < 0.5f + bias * 0.25f)
            {
                return Random.value < 0.5f ? 0 : 2; // 0° or 180°
            }
            else
            {
                return Random.value < 0.5f ? 1 : 3; // 90° or 270°
            }
        }
        else
        {
            // Bias toward 90° and 270°
            float bias = (rotationBias - 0.5f) * 2f; // 0 to 1
            if (rand < 0.5f + bias * 0.25f)
            {
                return Random.value < 0.5f ? 1 : 3; // 90° or 270°
            }
            else
            {
                return Random.value < 0.5f ? 0 : 2; // 0° or 180°
            }
        }
    }
    
    /// <summary>
    /// Find a valid grid position using clustering factor
    /// </summary>
    private Vector2Int FindValidGridPosition(int widthCells, int heightCells)
    {
        int maxAttempts = 100;
        
        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            Vector2Int pos;
            
            // Use clustering factor to determine placement strategy
            if (placedGates.Count > 0 && Random.value < clusteringFactor)
            {
                // Try to place near existing gates
                GatePlacement nearGate = placedGates[Random.Range(0, placedGates.Count)];
                int offsetRange = Mathf.RoundToInt(5f * (1f - clusteringFactor));
                pos = new Vector2Int(
                    nearGate.gridPosition.x + Random.Range(-offsetRange, offsetRange + 1),
                    nearGate.gridPosition.y + Random.Range(-offsetRange, offsetRange + 1)
                );
            }
            else
            {
                // Random placement
                pos = new Vector2Int(
                    Random.Range(0, gridWidth - widthCells + 1),
                    Random.Range(0, gridHeight - heightCells + 1)
                );
            }
            
            // Check if position is valid
            if (IsPositionValid(pos, widthCells, heightCells))
            {
                return pos;
            }
        }
        
        return new Vector2Int(-1, -1); // No valid position found
    }
    
    /// <summary>
    /// Check if a position is valid (within bounds and not occupied)
    /// </summary>
    private bool IsPositionValid(Vector2Int pos, int widthCells, int heightCells)
    {
        // Check bounds
        if (pos.x < 0 || pos.y < 0 || 
            pos.x + widthCells > gridWidth || 
            pos.y + heightCells > gridHeight)
        {
            return false;
        }
        
        // Check occupancy
        for (int x = pos.x; x < pos.x + widthCells; x++)
        {
            for (int y = pos.y; y < pos.y + heightCells; y++)
            {
                if (occupiedCells[x, y])
                {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    /// <summary>
    /// Check if minimum distance constraint is satisfied
    /// </summary>
    private bool CheckMinimumDistance(Vector2Int pos)
    {
        if (minGateDistance <= 0) return true;
        
        foreach (var gate in placedGates)
        {
            float distance = Vector2Int.Distance(pos, gate.gridPosition);
            if (distance < minGateDistance)
            {
                return false;
            }
        }
        
        return true;
    }
    
    /// <summary>
    /// Mark cells as occupied or unoccupied
    /// </summary>
    private void MarkCellsOccupied(Vector2Int pos, int widthCells, int heightCells, bool occupied)
    {
        for (int x = pos.x; x < pos.x + widthCells; x++)
        {
            for (int y = pos.y; y < pos.y + heightCells; y++)
            {
                occupiedCells[x, y] = occupied;
            }
        }
    }
    
    /// <summary>
    /// Convert grid position to world position (centered on cells)
    /// </summary>
    private Vector3 GridToWorldPosition(Vector2Int gridPos, int widthCells, int heightCells)
    {
        // Center the grid around manager's position
        float gridWorldWidth = gridWidth * cellSize;
        float gridWorldHeight = gridHeight * cellSize;
        
        // Calculate position in grid-local space
        float x = (gridPos.x + widthCells * 0.5f) * cellSize - gridWorldWidth * 0.5f;
        float z = (gridPos.y + heightCells * 0.5f) * cellSize - gridWorldHeight * 0.5f;
        
        // Convert to world space relative to manager's position
        Vector3 localPos = new Vector3(x, 0, z);
        Vector3 worldPos = transform.position + localPos;
        
        // Apply offsets for pivot and manual adjustment
        worldPos += pivotOffset + positionOffset;
        
        return worldPos;
    }
    
    /// <summary>
    /// Visualize the grid in the editor
    /// </summary>
    private void OnDrawGizmos()
    {
        if (!showGridGizmos) return;
        
        float gridWorldWidth = gridWidth * cellSize;
        float gridWorldHeight = gridHeight * cellSize;
        Vector3 gridCenter = transform.position;
        
        // Draw grid lines
        Gizmos.color = Color.gray;
        
        for (int x = 0; x <= gridWidth; x++)
        {
            Vector3 start = gridCenter + new Vector3(
                x * cellSize - gridWorldWidth * 0.5f, 
                0, 
                -gridWorldHeight * 0.5f
            );
            Vector3 end = gridCenter + new Vector3(
                x * cellSize - gridWorldWidth * 0.5f, 
                0, 
                gridWorldHeight * 0.5f
            );
            Gizmos.DrawLine(start, end);
        }
        
        for (int y = 0; y <= gridHeight; y++)
        {
            Vector3 start = gridCenter + new Vector3(
                -gridWorldWidth * 0.5f, 
                0, 
                y * cellSize - gridWorldHeight * 0.5f
            );
            Vector3 end = gridCenter + new Vector3(
                gridWorldWidth * 0.5f, 
                0, 
                y * cellSize - gridWorldHeight * 0.5f
            );
            Gizmos.DrawLine(start, end);
        }
        
        // Draw occupied cells
        if (occupiedCells != null)
        {
            Gizmos.color = new Color(1, 0, 0, 0.3f);
            
            for (int x = 0; x < gridWidth; x++)
            {
                for (int y = 0; y < gridHeight; y++)
                {
                    if (occupiedCells[x, y])
                    {
                        Vector3 cellCenter = gridCenter + new Vector3(
                            (x + 0.5f) * cellSize - gridWorldWidth * 0.5f,
                            0.1f,
                            (y + 0.5f) * cellSize - gridWorldHeight * 0.5f
                        );
                        Gizmos.DrawCube(cellCenter, new Vector3(cellSize * 0.9f, 0.1f, cellSize * 0.9f));
                    }
                }
            }
        }
    }
    
    /// <summary>
    /// Apply physics material to all colliders on a gate instance
    /// </summary>
    private void ApplyPhysicsMaterialToGate(GameObject gate)
    {
        if (gatePhysicsMaterial == null && !gateTriggers)
        {
            return; // Nothing to apply
        }
        
        Collider[] colliders = gate.GetComponentsInChildren<Collider>();
        
        if (colliders.Length == 0)
        {
            Debug.LogWarning($"No colliders found on gate {gate.name}. " +
                           "Make sure 'Generate Colliders' is enabled in the model import settings.");
            return;
        }
        
        foreach (Collider col in colliders)
        {
            if (gatePhysicsMaterial != null)
            {
                col.material = gatePhysicsMaterial;
            }
            
            col.isTrigger = gateTriggers;
        }
    }
    
    /// <summary>
    /// Public method to regenerate maze with new seed
    /// </summary>
    public void RegenerateWithSeed(int newSeed)
    {
        seed = newSeed;
        GenerateMaze();
    }
    
    /// <summary>
    /// Public method to update semantic parameters
    /// </summary>
    public void UpdateSemanticParameters(float rotation, float clustering, float scale)
    {
        rotationBias = Mathf.Clamp01(rotation);
        clusteringFactor = Mathf.Clamp01(clustering);
        scaleVariation = Mathf.Clamp01(scale);
    }
}