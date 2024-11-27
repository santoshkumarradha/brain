from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, create_model


def create_dynamic_pydantic_model(
    schema_dict: Dict[str, Any], model_name: str = None, definitions: Dict = None
) -> type[BaseModel]:
    """Creates a Pydantic model dynamically from a schema dictionary"""
    if definitions is None:
        definitions = schema_dict.get("$defs", {})

    def resolve_ref(ref: str) -> Dict[str, Any]:
        if not ref.startswith("#/$defs/"):
            raise ValueError(f"Unsupported $ref format: {ref}")
        def_name = ref.split("/")[-1]
        return definitions[def_name]

    def _get_field_type(field_schema: Dict[str, Any], field_name: str) -> type:
        if "$ref" in field_schema:
            ref_schema = resolve_ref(field_schema["$ref"])
            ref_name = ref_schema.get("title", field_name)
            return create_dynamic_pydantic_model(ref_schema, ref_name, definitions)

        if field_schema.get("anyOf") or field_schema.get("oneOf"):
            union_types = []
            for sub_schema in field_schema.get("anyOf", []) or field_schema.get(
                "oneOf", []
            ):
                if "$ref" in sub_schema:
                    ref_schema = resolve_ref(sub_schema["$ref"])
                    ref_name = ref_schema.get("title", f"{field_name}Type")
                    union_types.append(
                        create_dynamic_pydantic_model(ref_schema, ref_name, definitions)
                    )
                else:
                    sub_type = _get_field_type(sub_schema, f"{field_name}Type")
                    union_types.append(sub_type)
            return Union[tuple(union_types)]

        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
        }

        field_type = field_schema.get("type")

        if field_type == "object":
            nested_model_name = (
                f"{model_name}_{field_name}" if model_name else field_name
            )
            return create_dynamic_pydantic_model(
                field_schema, nested_model_name, definitions
            )

        if field_type == "array":
            items_schema = field_schema.get("items", {})
            if "$ref" in items_schema:
                ref_schema = resolve_ref(items_schema["$ref"])
                ref_name = ref_schema.get("title", f"{field_name}Item")
                item_type = create_dynamic_pydantic_model(
                    ref_schema, ref_name, definitions
                )
                return List[item_type]
            elif items_schema.get("type") == "object":
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
                item_type = items_schema.get("type", "string")
                base_type = type_mapping.get(item_type, str)
                return List[base_type]

        return type_mapping.get(field_type, str)

    properties = schema_dict.get("properties", {})
    required = schema_dict.get("required", [])

    fields = {}
    for field_name, field_schema in properties.items():
        field_type = _get_field_type(field_schema, field_name)

        if field_name not in required:
            field_type = Optional[field_type]

        field = Field(
            description=field_schema.get("description", ""),
            title=field_schema.get("title", field_name),
        )

        fields[field_name] = (field_type, field)

    model_name = model_name or schema_dict.get("title", "DynamicModel")
    model = create_model(model_name, **fields)
    model.model_config = {
        "title": schema_dict.get("title", model_name),
        "description": schema_dict.get("description", ""),
    }

    return model
