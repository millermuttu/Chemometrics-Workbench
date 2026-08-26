/** Parameter forms built from `models.py`'s own JSON Schema.
 *
 * The rule from #47: where the form and the schema could drift, generate from
 * the schema rather than restating it. A field's type, its bounds, its enum
 * and its default all come from the served schema, so a form can neither
 * refuse what the backend allows nor allow what it refuses.
 *
 * What cannot be generated is the cross-field rules - an odd Savitzky-Golay
 * window, `polyorder` below it - because they live in `model_validator` and
 * JSON Schema has no way to say them. Those are checked by asking the server,
 * so the message a user reads is the model's own.
 */

export interface FieldSpec {
  name: string;
  title: string;
  description?: string;
  kind: "number" | "integer" | "enum" | "unsupported";
  options?: string[];
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  default?: unknown;
  optional: boolean;
}

export interface StepSpec {
  kind: string;
  title: string;
  fields: FieldSpec[];
}

interface JsonSchemaField {
  type?: string;
  title?: string;
  description?: string;
  const?: string;
  default?: unknown;
  enum?: string[];
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  anyOf?: JsonSchemaField[];
}

export interface StepSchema {
  $defs: Record<
    string,
    { title: string; properties: Record<string, JsonSchemaField & { const?: string }> }
  >;
}

/** Pydantic writes an optional field as `anyOf: [{...}, {type: "null"}]`. The
 * shape of the field is the half that is not null. */
function unwrap(field: JsonSchemaField): { field: JsonSchemaField; optional: boolean } {
  if (!field.anyOf) return { field, optional: false };
  const real = field.anyOf.find((option) => option.type !== "null");
  return { field: { ...field, ...real }, optional: field.anyOf.some((o) => o.type === "null") };
}

function toField(name: string, raw: JsonSchemaField): FieldSpec {
  const { field, optional } = unwrap(raw);
  const options = field.enum ?? (field.const ? [field.const] : undefined);
  const kind = options
    ? "enum"
    : field.type === "integer"
      ? "integer"
      : field.type === "number"
        ? "number"
        : "unsupported";
  return {
    name,
    title: field.title ?? name,
    description: field.description,
    kind,
    options,
    minimum: field.minimum,
    maximum: field.maximum,
    exclusiveMinimum: field.exclusiveMinimum,
    exclusiveMaximum: field.exclusiveMaximum,
    default: field.default,
    optional,
  };
}

export function stepSpecs(schema: StepSchema): StepSpec[] {
  return Object.values(schema.$defs).map((definition) => ({
    kind: definition.properties.kind.const ?? definition.title.toLowerCase(),
    title: definition.title,
    // `kind` is the discriminator, not a parameter: it names the step and is
    // never edited.
    fields: Object.entries(definition.properties)
      .filter(([name]) => name !== "kind")
      .map(([name, field]) => toField(name, field)),
  }));
}

export function specFor(schema: StepSchema | undefined, kind: string): StepSpec | undefined {
  return schema ? stepSpecs(schema).find((spec) => spec.kind === kind) : undefined;
}

/** The bounds a single field can be checked against without asking anyone.
 * Returns the message to show, or null when the value is allowed. */
export function checkBounds(field: FieldSpec, value: number | string | null): string | null {
  if (value === null || value === "") {
    return field.optional ? null : `${field.title} is required`;
  }
  if (field.kind === "enum") {
    return field.options?.includes(String(value))
      ? null
      : `${field.title} must be one of ${field.options?.join(", ")}`;
  }
  const number = Number(value);
  if (Number.isNaN(number)) return `${field.title} must be a number`;
  if (field.kind === "integer" && !Number.isInteger(number)) {
    return `${field.title} must be a whole number`;
  }
  if (field.minimum !== undefined && number < field.minimum) {
    return `${field.title} must be at least ${field.minimum}`;
  }
  if (field.maximum !== undefined && number > field.maximum) {
    return `${field.title} must be at most ${field.maximum}`;
  }
  if (field.exclusiveMinimum !== undefined && number <= field.exclusiveMinimum) {
    return `${field.title} must be greater than ${field.exclusiveMinimum}`;
  }
  if (field.exclusiveMaximum !== undefined && number >= field.exclusiveMaximum) {
    return `${field.title} must be less than ${field.exclusiveMaximum}`;
  }
  return null;
}
