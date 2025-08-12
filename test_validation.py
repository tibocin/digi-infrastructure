from pcs.api.v1.contexts import ContextTypeCreate

print("Testing validation...")
try:
    ContextTypeCreate(name="test", type_enum="custom", supports_vectors=True)
    print("No error raised - validation not working")
except Exception as e:
    print("Error:", e)
