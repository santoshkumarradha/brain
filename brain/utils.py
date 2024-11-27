from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, create_model


def create_dynamic_pydantic_model(
    schema_dict: Dict[str, Any], model_name: str = None, definitions: Dict = None
) -> type[BaseModel]:
    """
    Creates a Pydantic model dynamically from a schema dictionary, supporting $ref references and $defs.

    Args:
        schema_dict: Dictionary containing the schema definition
        model_name: Optional name for the model (used for nested models)
        definitions: Dictionary of schema definitions (used internally for resolving $refs)

    Returns:
        A dynamically created Pydantic model class
    """
    # Initialize definitions on first call
    if definitions is None:
        definitions = schema_dict.get("$defs", {})

    def resolve_ref(ref: str) -> Dict[str, Any]:
        """Resolve a $ref reference to its schema definition"""
        if not ref.startswith("#/$defs/"):
            raise ValueError(f"Unsupported $ref format: {ref}")
        def_name = ref.split("/")[-1]
        return definitions[def_name]

    def _get_field_type(field_schema: Dict[str, Any], field_name: str) -> type:
        """Helper function to determine field type, handling refs and nested objects"""
        # Handle $ref references
        if "$ref" in field_schema:
            ref_schema = resolve_ref(field_schema["$ref"])
            ref_name = ref_schema.get("title", field_name)
            return create_dynamic_pydantic_model(ref_schema, ref_name, definitions)

        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
        }

        field_type = field_schema.get("type")

        if field_type == "object":
            # Recursively create nested model
            nested_model_name = (
                f"{model_name}_{field_name}" if model_name else field_name
            )
            return create_dynamic_pydantic_model(
                field_schema, nested_model_name, definitions
            )

        if field_type == "array":
            items_schema = field_schema.get("items", {})
            if "$ref" in items_schema:
                # Handle array of referenced types
                ref_schema = resolve_ref(items_schema["$ref"])
                ref_name = ref_schema.get("title", f"{field_name}Item")
                item_type = create_dynamic_pydantic_model(
                    ref_schema, ref_name, definitions
                )
                return List[item_type]
            elif items_schema.get("type") == "object":
                # Handle array of objects
                nested_model_name = (
                    f"{model_name}_{field_name}Item"
                    if model_name
                    else f"{field_name}Item"
                )
                nested_model = create_dynamic_pydantic_model(
                    items_schema, nested_model_name, definitions
                )
                return List[nested_model]
            else:
                # Handle array of primitive types
                item_type = items_schema.get("type", "string")
                base_type = type_mapping.get(item_type, str)
                return List[base_type]

        return type_mapping.get(field_type, str)

    # Extract properties from schema
    properties = schema_dict.get("properties", {})
    required = schema_dict.get("required", [])

    # Prepare field definitions for create_model
    fields = {}
    for field_name, field_schema in properties.items():
        field_type = _get_field_type(field_schema, field_name)

        # Make field optional if not required
        if field_name not in required:
            field_type = Optional[field_type]

        # Create field with description and other metadata, but no default
        field = Field(
            description=field_schema.get("description", ""),
            title=field_schema.get("title", field_name),
        )

        fields[field_name] = (field_type, field)

    # Use provided model name or fall back to schema title or default
    model_name = model_name or schema_dict.get("title", "DynamicModel")

    # Create model with schema configuration
    model = create_model(model_name, **fields)

    # Add schema configuration
    model.model_config = {
        "title": schema_dict.get("title", model_name),
        "description": schema_dict.get("description", ""),
    }

    return model
