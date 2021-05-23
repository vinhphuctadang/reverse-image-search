# FAQs and troubleshoot

1. 

```
Unhandled exception. System.MissingMethodException: Method not found: 'System.Reflection.Emit.AssemblyBuilder System.AppDomain.DefineDynamicAssembly(System.Reflection.AssemblyName, System.Reflection.Emit.AssemblyBuilderAccess)'.
   at Python.Runtime.CodeGenerator..ctor()
   at Python.Runtime.DelegateManager..ctor()
   at Python.Runtime.PythonEngine.Initialize(IEnumerable`1 args, Boolean setSysArgv, Boolean initSigs)
   at Python.Runtime.PythonEngine.Initialize(Boolean setSysArgv, Boolean initSigs)
   at Python.Runtime.PythonEngine.Initialize()
   at Python.Runtime.Py.GIL()
```

**Solution**: Should change TargetFramework to net47, below it maybe ok, but I didn't test