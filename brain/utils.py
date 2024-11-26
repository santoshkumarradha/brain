from pydantic import BaseModel, create_model


def schema_to_pydantic_class(schema: dict) -> type:
    """
    Convert a JSON schema into a Pydantic BaseModel class.

    Args:
        schema (dict): The JSON schema to convert.

    Returns:
        type: A dynamically generated Pydantic BaseModel class.
    """
    if schema.get("type") != "object":
        raise ValueError("Only 'object' schemas can be converted to Pydantic classes.")

    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    # Dynamically build the fields for the Pydantic model
    fields = {}
    for field_name, field_properties in properties.items():
        field_type = field_properties.get("type")
        python_type = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }.get(
            field_type, str
        )  # Default to `str` if type is unknown

        # Mark field as required or optional
        if field_name in required_fields:
            fields[field_name] = (python_type, ...)
        else:
            fields[field_name] = (python_type, None)  # Optional field

    # Create the Pydantic model class
    model_name = schema.get("title", "DynamicModel")
    return create_model(model_name, **fields)
