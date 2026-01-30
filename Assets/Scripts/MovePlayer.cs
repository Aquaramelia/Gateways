// Based on https://docs.unity3d.com/6000.3/Documentation/ScriptReference/CharacterController.Move.html

using UnityEngine;
using UnityEngine.InputSystem;

public class MovePlayer : MonoBehaviour
{
    [Header("Movement")]
    [SerializeField] private float moveSpeed = 5f;
    [SerializeField] private float rotationSpeed = 10f;
    
    private Vector2 moveInput;
    private Vector2 lookInput;
    private CharacterController characterController;
    private Camera mainCamera;
    private float cameraPitch = 0f;

    void Start()
    {
        // Add CharacterController if not present
        characterController = GetComponent<CharacterController>();
        if (characterController == null)
        {
            characterController = gameObject.AddComponent<CharacterController>();
        }
        
        mainCamera = Camera.main;
        
        // Lock and hide cursor
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;
    }

    void Update()
    {
        Move();
        Look();
    }

    // Called by Player Input component
    public void OnMove(InputValue value)
    {
        moveInput = value.Get<Vector2>();
    }

    // Called by Player Input component
    public void OnLook(InputValue value)
    {
        lookInput = value.Get<Vector2>();
    }

    private void Move()
    {
        if (moveInput == Vector2.zero)
        {
            // Still apply gravity so the player remains grounded
            characterController.Move(Vector3.down * 9.81f * Time.deltaTime);
            return;
        }

        // Get movement direction relative to camera
        Vector3 forward = mainCamera.transform.forward;
        Vector3 right = mainCamera.transform.right;
        
        // Flatten directions (no vertical component)
        forward.y = 0;
        right.y = 0;
        forward.Normalize();
        right.Normalize();
        
        // Calculate move direction
        Vector3 moveDirection = (forward * moveInput.y + right * moveInput.x).normalized;
        
        // Apply movement
        Vector3 move = moveDirection * moveSpeed * Time.deltaTime;
        
        // Add gravity
        move.y = -9.81f * Time.deltaTime;
        
        characterController.Move(move);
    }

    private void Look()
    {
        if (lookInput == Vector2.zero)
        return;

        // Rotate player horizontally
        transform.Rotate(Vector3.up * lookInput.x * rotationSpeed * Time.deltaTime);
        
        // Rotate camera vertically
        cameraPitch -= lookInput.y * rotationSpeed * Time.deltaTime;
        cameraPitch = Mathf.Clamp(cameraPitch, -90f, 90f);
        
        if (mainCamera != null)
        {
            mainCamera.transform.localRotation = Quaternion.Euler(cameraPitch, 0, 0);
        }
    }
}
