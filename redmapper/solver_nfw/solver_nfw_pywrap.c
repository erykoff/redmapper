#include <Python.h>
#include "solver_nfw.h"
#include <numpy/arrayobject.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


struct SolverObject {
    PyObject_HEAD
    struct solver* solver;
    int allocated;
};
static void
SolverObject_dealloc(struct SolverObject* self)
{
    if (self->allocated) {
	free(self->solver);
	self->allocated = 0;
    }

#if PY_MAJOR_VERSION >= 3
    Py_TYPE(self)->tp_free((PyObject*)self);
#else
    self->ob_type->tp_free((PyObject*)self);
#endif
}

static int
SolverObject_init(struct SolverObject* self, PyObject *args)
{
    double r0;
    double beta;
    PyArrayObject *ucounts_obj = NULL;
    PyArrayObject *bcounts_obj = NULL;
    PyArrayObject *r_obj = NULL;
    PyArrayObject *w_obj = NULL;
    PyArrayObject *cpars_obj = NULL;
    double rsig;

    self->allocated = 0;

    if (!PyArg_ParseTuple(args,
			  (char*)"ddOOOOOd",
			  &r0,
			  &beta,
			  &ucounts_obj,
			  &bcounts_obj,
			  &r_obj,
			  &w_obj,
			  &cpars_obj,
			  &rsig)) {
	PyErr_SetString(PyExc_RuntimeError,"Failed to parse init");
	return -1;
    }

    if ((self->solver = calloc(1,sizeof(struct solver))) == NULL) {
	PyErr_SetString(PyExc_RuntimeError,"Failed to allocate struct solver");
	return -1;
    }
    self->allocated = 1;

    self->solver->ucounts = (double *) PyArray_DATA(ucounts_obj);
    self->solver->bcounts = (double *) PyArray_DATA(bcounts_obj);
    self->solver->r = (double *) PyArray_DATA(r_obj);
    self->solver->w = (double *) PyArray_DATA(w_obj);
    self->solver->cpars = (double *) PyArray_DATA(cpars_obj);

    self->solver->ngal = (long) PyArray_DIM(ucounts_obj, 0);

    self->solver->r0 = r0;
    self->solver->beta = beta;
    self->solver->rsig = rsig;

    if (PyArray_DIM(cpars_obj, 0) != CPAR_NTERMS) {
	PyErr_SetString(PyExc_ValueError, "cpars with wrong number of terms");
	return -1;
    }

    return 0;
}

static PyObject*
SolverObject_repr(struct SolverObject* self) {
    char repr[256];
    sprintf(repr, "Solver Object");
#if PY_MAJOR_VERSION >= 3
    return Py_BuildValue("y",repr);
#else
    return Py_BuildValue("s",repr);
#endif
}

PyObject* SolverObject_solver_nfw(const struct SolverObject* self, PyObject *args)
{
    npy_intp dims[1];
    PyObject* lam_obj = NULL;
    PyObject* rlam_obj = NULL;
    PyObject* p_obj = NULL;
    PyObject* wt_obj = NULL;
    PyObject* thetar_obj = NULL;

    dims[0] = 1;
    lam_obj = PyArray_ZEROS(0, dims, NPY_DOUBLE, 0);
    rlam_obj = PyArray_ZEROS(0, dims, NPY_DOUBLE, 0);

    dims[0] = self->solver->ngal;
    p_obj = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    wt_obj = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    thetar_obj = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    // do the work

    // use the allocated arrays...
    self->solver->p = (double *) PyArray_DATA((PyArrayObject*)p_obj);
    self->solver->wt = (double *) PyArray_DATA((PyArrayObject*)wt_obj);
    self->solver->lambda = (double *) PyArray_DATA((PyArrayObject*)lam_obj);
    self->solver->rlambda = (double *) PyArray_DATA((PyArrayObject*)rlam_obj);
    self->solver->theta_r = (double *) PyArray_DATA((PyArrayObject*)thetar_obj);

    solver_nfw(self->solver->r0, self->solver->beta, self->solver->ngal,
	       self->solver->ucounts, self->solver->bcounts, self->solver->r,
	       self->solver->w, self->solver->lambda, self->solver->p, self->solver->wt,
               self->solver->rlambda, self->solver->theta_r,
	       TOL_DEFAULT, self->solver->cpars, self->solver->rsig);

    // this needs to return the tuple.

    PyObject* retval = PyTuple_New(5);
    PyTuple_SET_ITEM(retval, 0, PyArray_Return((PyArrayObject *)lam_obj));
    PyTuple_SET_ITEM(retval, 1, PyArray_Return((PyArrayObject *)p_obj));
    PyTuple_SET_ITEM(retval, 2, PyArray_Return((PyArrayObject *)wt_obj));
    PyTuple_SET_ITEM(retval, 3, PyArray_Return((PyArrayObject *)rlam_obj));
    PyTuple_SET_ITEM(retval, 4, PyArray_Return((PyArrayObject *)thetar_obj));

    return retval;
}

static PyMethodDef SolverObject_methods[] = {
    {"solve_nfw", (PyCFunction)SolverObject_solver_nfw, METH_NOARGS, "solve_nfw()"},
    {NULL} /* Sentinel */
};

static PyTypeObject PySolverType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_solver_nfw_pywrap.Solver",             /*tp_name*/
    sizeof(struct SolverObject), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)SolverObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    (reprfunc)SolverObject_repr,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "Solver Class",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    SolverObject_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    //0,     /* tp_init */
    (initproc)SolverObject_init,      /* tp_init */
    0,                         /* tp_alloc */
    //PyPSFExObject_new,                 /* tp_new */
    PyType_GenericNew,                 /* tp_new */
};


static PyMethodDef Solver_module_methods[] = {
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_solver_nfw_pywrap",      /* m_name */
    "Solver (w/ NFW) method",  /* m_doc */
    -1,                        /* m_size */
    Solver_module_methods,    /* m_methods */
    NULL,                      /* m_reload */
    NULL,                      /* m_traverse */
    NULL,                      /* m_clear */
    NULL,                      /* m_free */
};
#endif


#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit__solver_nfw_pywrap(void)
#else
init_solver_nfw_pywrap(void)
#endif
{
    PyObject* m;

    PySolverType.tp_new = PyType_GenericNew;

#if PY_MAJOR_VERSION >= 3
    if (PyType_Ready(&PySolverType) < 0) {
        return NULL;
    }

    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else

    if (PyType_Ready(&PySolverType) < 0)
	return;

    m = Py_InitModule3("_solver_nfw_pywrap", Solver_module_methods, "Solver (w/ NFW) method");
#endif

    Py_INCREF(&PySolverType);
    PyModule_AddObject(m, "Solver", (PyObject *) &PySolverType);

    import_array();
#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
