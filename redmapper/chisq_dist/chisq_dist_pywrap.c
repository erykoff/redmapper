#include <Python.h>
#include "chisq_dist.h"
#include "stdio.h"
#include <numpy/arrayobject.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

struct ChisqDistObject {
    PyObject_HEAD
    struct chisq_dist* chisq_dist;
    int allocated;
};
static void
ChisqDistObject_dealloc(struct ChisqDistObject* self)
{
    if (self->allocated) {
	free(self->chisq_dist);
	self->allocated = 0;
    }

#if PY_MAJOR_VERSION >= 3
    Py_TYPE(self)->tp_free((PyObject*)self);
#else
    self->ob_type->tp_free((PyObject*)self);
#endif
}

static int
ChisqDistObject_init(struct ChisqDistObject* self, PyObject* args)
{
    long mode;
    long ngal;
    long nz;
    long ncol;
    PyArrayObject *covmat_obj = NULL;
    PyArrayObject *c_obj = NULL;
    PyArrayObject *slope_obj = NULL;
    PyArrayObject *pivotmag_obj = NULL;
    PyArrayObject *refmag_obj = NULL;
    PyArrayObject *refmagerr_obj = NULL;
    PyArrayObject *magerr_obj = NULL;
    PyArrayObject *color_obj = NULL;
    PyArrayObject *lupcorr_obj = NULL;

    // debug
    // int i,j,k,stride;

    self->allocated = 0;

    if (!PyArg_ParseTuple(args,
			  (char*)"iiiiOOOOOOOOO",
			  &mode,
			  &ngal,
			  &nz,
			  &ncol,
			  &covmat_obj,
			  &c_obj,
			  &slope_obj,
			  &pivotmag_obj,
			  &refmag_obj,
			  &refmagerr_obj,
			  &magerr_obj,
			  &color_obj,
			  &lupcorr_obj)){
	PyErr_SetString(PyExc_RuntimeError,"Failed to parse init");
	return -1;
    }

    if ((self->chisq_dist = calloc(1,sizeof(struct chisq_dist))) == NULL) {
	PyErr_SetString(PyExc_RuntimeError,"Failed to allocate struct chisq_dist");
	return -1;
    }
    self->allocated = 1;

    self->chisq_dist->mode = mode;
    self->chisq_dist->ngal = ngal;
    self->chisq_dist->nz = nz;
    self->chisq_dist->ncol = ncol;
    if (PyArray_TYPE(covmat_obj) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_ValueError, "covmat must be of type float64");
        return -1;
    }
    self->chisq_dist->covmat = (double *) PyArray_DATA(covmat_obj);
    if (PyArray_TYPE(c_obj) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_ValueError, "c must be of type float64");
        return -1;
    }
    self->chisq_dist->c = (double *) PyArray_DATA(c_obj);
    if (PyArray_TYPE(slope_obj) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_ValueError, "slope must be of type float64");
        return -1;
    }
    self->chisq_dist->slope = (double *) PyArray_DATA(slope_obj);
    if (PyArray_TYPE(pivotmag_obj) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_ValueError, "pivotmag must be of type float64");
        return -1;
    }
    self->chisq_dist->pivotmag = (double *) PyArray_DATA(pivotmag_obj);
    if (PyArray_TYPE(refmag_obj) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_ValueError, "refmag must be of type float64");
        return -1;
    }
    self->chisq_dist->refmag = (double *) PyArray_DATA(refmag_obj);
    if (PyArray_TYPE(refmagerr_obj) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_ValueError, "refmagerr must be of type float64");
        return -1;
    }
    self->chisq_dist->refmagerr = (double *) PyArray_DATA(refmagerr_obj);
    if (PyArray_TYPE(magerr_obj) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_ValueError, "magerr must be of type float64");
        return -1;
    }
    self->chisq_dist->magerr = (double *) PyArray_DATA(magerr_obj);
    if (PyArray_TYPE(color_obj) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_ValueError, "color must be of type float64");
        return -1;
    }
    self->chisq_dist->color = (double *) PyArray_DATA(color_obj);
    if (PyArray_TYPE(lupcorr_obj) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_ValueError, "lupcorr must be of type float64");
        return -1;
    }
    self->chisq_dist->lupcorr = (double *) PyArray_DATA(lupcorr_obj);

    if (self->chisq_dist->mode == 0) {
	self->chisq_dist->ncalc = ngal;
    } else {
	self->chisq_dist->ncalc = nz;
    }

    self->chisq_dist->sigint = SIGINT_DEFAULT;

    return 0;
}

static PyObject*
ChisqDistObject_repr(struct ChisqDistObject* self) {
    char repr[256];
    sprintf(repr, "Chisq Dist Object");
#if PY_MAJOR_VERSION >= 3
    return Py_BuildValue("y",repr);
#else
    return Py_BuildValue("s",repr);
#endif
}

PyObject* ChisqDistObject_compute(const struct ChisqDistObject* self, PyObject *args)
{
    npy_intp dims[1];
    PyObject* chisq_obj = NULL;
    double *chisq;
    int do_chisq, nophotoerr;

    dims[0] = self->chisq_dist->ncalc;
    chisq_obj = PyArray_ZEROS(1, dims, NPY_FLOAT64, 0);

    chisq = (double *) PyArray_DATA((PyArrayObject*)chisq_obj);

    // parse the args
    if (!PyArg_ParseTuple(args,
			  (char*)"ii",
			  &do_chisq,
			  &nophotoerr)) {
	PyErr_SetString(PyExc_RuntimeError,"Failed to parse args");
	return PyArray_Return((PyArrayObject *) chisq_obj);
    }

    // and do the work.
    chisq_dist(self->chisq_dist->mode, do_chisq, nophotoerr, self->chisq_dist->ncalc,
	       self->chisq_dist->ncol, self->chisq_dist->covmat, self->chisq_dist->c,
	       self->chisq_dist->slope, self->chisq_dist->pivotmag, self->chisq_dist->refmag,
	       self->chisq_dist->refmagerr, self->chisq_dist->magerr, self->chisq_dist->color,
	       self->chisq_dist->lupcorr, chisq, self->chisq_dist->sigint);

    return PyArray_Return((PyArrayObject *) chisq_obj);
}

static PyMethodDef ChisqDistObject_methods[] = {
    {"compute", (PyCFunction)ChisqDistObject_compute, METH_VARARGS, "compute()"},
    {NULL} /* Sentinel */
};

static PyTypeObject PyChisqDistType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_chisq_dist_pywrap.ChisqDist",             /*tp_name*/
    sizeof(struct ChisqDistObject), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)ChisqDistObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    (reprfunc)ChisqDistObject_repr,                         /*tp_repr*/
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
    "ChisqDist Class",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    ChisqDistObject_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    //0,     /* tp_init */
    (initproc)ChisqDistObject_init,      /* tp_init */
    0,                         /* tp_alloc */
    //PyPSFExObject_new,                 /* tp_new */
    PyType_GenericNew,                 /* tp_new */
};

static PyMethodDef ChisqDist_module_methods[] = {
    {NULL}  /* Sentinel */
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_chisq_dist_pywrap",      /* m_name */
    "ChisqDist (with color)",  /* m_doc */
    -1,                        /* m_size */
    ChisqDist_module_methods,    /* m_methods */
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
PyInit__chisq_dist_pywrap(void)
#else
init_chisq_dist_pywrap(void)
#endif
{
    PyObject* m;

    PyChisqDistType.tp_new = PyType_GenericNew;

#if PY_MAJOR_VERSION >= 3
    if (PyType_Ready(&PyChisqDistType) < 0) {
        return NULL;
    }

    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else
    if (PyType_Ready(&PyChisqDistType) < 0)
	return;

    m = Py_InitModule3("_chisq_dist_pywrap", ChisqDist_module_methods, "ChisqDist (with color)");
    if (m==NULL) {
        return;
    }
#endif

    Py_INCREF(&PyChisqDistType);
    PyModule_AddObject(m, "ChisqDist", (PyObject *) &PyChisqDistType);

    import_array();

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}

