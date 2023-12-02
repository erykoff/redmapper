"""Generic object and catalog classes for redmapper.

This file contains classes to describe numpy wrappers which makes things look
like the old idl code (for good or ill).  It also makes some things a bit more
readable.
"""

import fitsio
import esutil as eu
import numpy as np
import itertools


class DataObject(object):
    """
    Generic DataObject class.

    This class wraps numpy ndarrays for more convenient use, and contains useful methods to saving/reading from fits files.
    """

    def __init__(self, *arrays):
        """
        Instantiate a DataObject from an arbitrary number of numpy ndarrays

        Each ndarray can have an arbitrary number of fields.  Each ndarray must
        have the same number of rows, and field names must be unique.

        Parameters
        ----------
        *arrays: `np.ndarray` parameters
        """
        for array in arrays:
            self._lower_array(array)

        if len(arrays) == 1:
            self._ndarray = arrays[0]
        else:
            self._ndarray = self._merge_arrays(arrays)

    @classmethod
    def from_fits_file(cls, filename, ext=1, rows=None):
        """
        Construct a DataObject from a fits file.

        Parameters
        ----------
        filename: `string`
           Filename to read
        ext: `int` or `string`, optional
           Extension number or name.  Default is 1.
        rows: `np.array`, optional
           Row indices to read.  Default is None (read all rows).
        """
        array = fitsio.read(filename, ext=ext, rows=rows, lower=True, trim_strings=True)
        return cls(array)

    @classmethod
    def from_fits_ext(cls, fits_ext):
        """
        Construct a DataObject from a fitsio fits extension

        Parameters
        ----------
        fits_ext: `fitsio.fitslib.TableHDU`
           Fits extension table
        """
        array = fits_ext.read(upper=True)
        return cls(array)

    @classmethod
    def zeros(cls, size, dtype):
        """
        Construct a DataObject filled with all 0s.

        Parameters
        ----------
        size: `int`
           Size of DataObject
        dtype: data-type
           `np.dtype` description
        """
        return cls(np.zeros(size, dtype=dtype))

    def __getattr__(self, attr):
        try:
            return self._ndarray[attr.lower()]
        except:
            return object.__getattribute__(self, attr)

    def __setattr__(self, attr, val):
        if attr == '_ndarray':
            object.__setattr__(self, attr, val)
        elif attr.lower() in self._ndarray.dtype.names:
            if hasattr(val, "__len__") and len(val) == 1:
                self._ndarray[attr.lower()] = val[0]
            else:
                self._ndarray[attr.lower()] = val
        else:
            object.__setattr__(self, attr, val)

    @property
    def dtype(self):
        """
        Return the numpy dtype associated with the DataObject.
        """
        return self._ndarray.dtype

    def add_fields(self, newdtype):
        """
        Add new fields to an existing DataObject (all filled with zeros).
        Modifications are done in-place.

        Parameters
        ----------
        newdtype: data-type
           `np.dtype` description
        """
        array = np.zeros(self._ndarray.size, newdtype)
        self._lower_array(array)
        self._ndarray = self._merge_arrays([self._ndarray, array])

    def to_fits_file(self, filename, clobber=False, header=None, extname=None, indices=None):
        """
        Save DataObject to a fits file.

        Parameters
        ----------
        filename: `str`
           Filename for output
        clobber: `bool`, optional
           Clobber existing file?  Default is False.
        header: `fitsio.FITSHDR`, optional
           Header to put on output file.  Default is None.
        extname: `str`, optional
           Extension name to put on output structure.  Default is None.
        indices: `np.array`, optional
           Indices of rows to output.  Default is None (output all).
        """
        if self._ndarray.size == 1:
            temp_array = np.zeros(1, dtype=self._ndarray.dtype)
            temp_array[0] = self._ndarray
            fitsio.write(filename, temp_array, clobber=clobber, header=header, extname=extname)
        else:
            if indices is None:
                fitsio.write(filename, self._ndarray, clobber=clobber, header=header, extname=extname)
            else:
                fitsio.write(filename, self._ndarray[indices], clobber=clobber, header=header, extname=extname)

    def _merge_arrays(self, arrays):
        """
        Internal method to merge multiple `np.ndarray`s relatively efficiently.

        Parameters
        ----------
        arrays: `list` of `np.ndarray`
           Arrays to merge.

        Returns
        -------
        merged_array: `np.ndarray`
           Merged array
        """

        dtype = None
        for array in arrays:
            if dtype is None:
                # First array
                dtype = array.dtype.descr
                names = array.dtype.names
                size = array.size
            else:
                # Not the first array
                # Check the size is the same
                if array.size != size:
                    raise ValueError("Cannot merge arrays of different length")
                # Check that we don't have any duplicate names
                for name in array.dtype.names:
                    if name in names:
                        raise ValueError("Cannot merge arrays with duplicate names (%s)" % (name))
                # Extend the dtype
                dtype.extend(array.dtype.descr)

        # Now we have what we need
        merged_array = np.zeros(size, dtype=dtype)

        # Copy in the arrays to the merged array, column by column
        for array in arrays:
            for name in array.dtype.names:
                merged_array[name] = array[name]

        return merged_array

    def _lower_array(self, array):
        """
        Change all array names to lower case.

        Parameters
        ----------
        array: `np.array`
        """
        names = list(array.dtype.names)
        array.dtype.names = [n.lower() for n in names]

    def __repr__(self):
        # return the representation of the underlying array
        return repr(self._ndarray)

    def __str__(self):
        # return the string of the underlying array
        return str(self._ndarray)

    def __dir__(self):
        # lower case list of all the available variables
        # also need to know original __dir__!
        return sorted(set(
                dir(type(self)) +
                self.__dict__.keys() +
                [x.lower() for x in self._ndarray.dtype.names]))


class Entry(DataObject):
    """
    An Entry is an extension of a DataObject.  It is intended to be used as a
    single Entry in a Catalog object.
    """
    def __init__(self, array):
        """
        Instantiate an Entry.

        Parameters
        ----------
        array: `np.ndarray` of length 1
        """
        if array.size != 1:
            raise ValueError("Input array must have length one.")
        # If this is an array of length 1, we want it to be a scalar-ish
        if len(array.shape) == 0:
            super(Entry, self).__init__(array)
        else:
            super(Entry, self).__init__(array[0])

    # @classmethod
    # def from_dict(cls, dict): pass

    def add_fields(self, newdtype):
        """
        Add fields to an Entry.

        Parameters
        ----------
        newdtype: data-type
           `np.dtype` description
        """
        array = np.zeros(self._ndarray.size, newdtype)
        self._lower_array(array)
        self._ndarray = self._merge_arrays([self._ndarray, array])[0]

    def __getattr__(self, attr):
        try:
            return self._ndarray[attr.lower()]
        except:
            return object.__getattribute__(self, attr)


class Catalog(DataObject):
    """
    A Catalog is an extension of a DataObject.  It can be decomposed into
    individual Entry objects.  This class is used to describe catalogs of all
    sorts, including galaxy catalogs, cluster catalogs, and other ndarrays.
    """

    entry_class = Entry

    @property
    def size(self):
        """
        Return the size of the Catalog.
        """
        return self._ndarray.size

    def append(self, append_cat):
        """
        Append a number of rows to the catalog, in-place.

        Parameters
        ----------
        append_cat: `redmapper.Catalog` or `np.ndarray`
           Catalog to append
        """
        if isinstance(append_cat, Catalog):
            self._ndarray = np.append(self._ndarray, append_cat._ndarray)
        else:
            self._ndarray = np.append(self._ndarray, append_cat)

    def extend(self, n_new):
        """
        Extend catalog with zero-filled rows, in-place.

        Parameters
        ----------
        n_new: `int`
           Number of new rows to append
        """
        temp = np.zeros(n_new, dtype=self._ndarray.dtype)
        self._ndarray = np.append(self._ndarray, temp)

    def __len__(self): return self.size

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.entry_class(self._ndarray.__getitem__(key))
        return type(self)(self._ndarray.__getitem__(key))

    def __setitem__(self, key, val):
        self._ndarray.__setitem__(key, val)

