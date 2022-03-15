'''
NAME
    A simple NetCDF wrapper Python
PURPOSE
    To demonstrate how to read and write data with NetCDF files using
    a NetCDF file from the NCEP/NCAR Reanalysis.
    Plotting using Matplotlib and Basemap is also shown.
REFERENCES
    netcdf4-python -- http://code.google.com/p/netcdf4-python/
    NCEP/NCAR Reanalysis -- Kalnay et al. 1996
        http://dx.doi.org/10.1175/1520-0477(1996)077<0437:TNYRP>2.0.CO;2
'''
import datetime as dt
import numpy as np
# http://code.google.com/p/netcdf4-python/ or https://github.com/Unidata/netcdf4-python (Python/numpy interface to the netCDF C library.)
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from collections import defaultdict
from src.common.log import logger


class ncdump():
    '''
    ncdump collects dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Init parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object

    Class fields
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_variables : list
        A Python list of the NetCDF file variables
    '''

    def __init__(self, nc_fid):
        # NetCDF global attributes
        attrs = nc_fid.ncattrs()
        self.nc_attrs = defaultdict()
        for attr in attrs:
            self.nc_attrs[attr] = repr(nc_fid.getncattr(attr))

        # Variable information.
        # list of nc variables
        variables = nc_fid.variables
        self.nc_variables = defaultdict()
        for var_name, var_object in variables.items():
            self.nc_variables[var_name] = defaultdict()
            for attr in var_object.ncattrs():
                self.nc_variables[var_name][attr] = var_object.getncattr(attr)

        # Dimension shape information.
        # list of nc dimensions
        dims = nc_fid.dimensions
        self.nc_dims = defaultdict()
        for dim_name, dim_object in dims.items():
            self.nc_dims[dim_name] = dim_object.size

    def __print_ncattr(self, key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            for attr, value in self.nc_variables[key].items():
                logger.info(f'{attr}, {value}')
        except KeyError:
            logger.info(f"WARNING: {key} does not contain variable attributes")

    def __print_global_ncattr(self):
        logger.info("--------NetCDF Global Attributes:--------")
        for key, value in self.nc_attrs.items():
            try:
                logger.info(f'{key}, {value}')
            except Exception as e:
                logger.error(f'Exception: {e}')

    def __print_ncdims(self):
        logger.info("--------NetCDF dimension information:--------")
        for dim_name, dim_size in self.nc_dims.items():
            try:
                logger.info(f"Name: {dim_name}")
                logger.info(f"size: {dim_size}")
                self.__print_ncattr(dim_name)
            except Exception as e:
                logger.error(f'Exception: {e}')

    def __print_ncvars(self):
        logger.info("--------NetCDF variable information:--------")
        for var in self.nc_variables:
            if var not in self.nc_dims:
                try:
                    logger.info(f'Name: {var}')
                    self.__print_ncattr(var)
                except Exception as e:
                    logger.error(f'Exception: {e}')

    def print_all(self):
        self.__print_global_ncattr()
        self.__print_ncdims()
        self.__print_ncvars()


def parse_nc_file(nc_file_path):
    '''
    A method to extract the metadata of a .nc file, leveraging the ncdump class

    Input: nc_file_path (the path to the .nc file - duh)

    Output: a ncdump object (cf. ncdump class documentation)
    '''
    # Dataset is the class behavior to open the file
    nc_fid = Dataset(nc_file_path, 'r')
    # and create an instance of the ncCDF4 class
    nc_dump = ncdump(nc_fid)

    # Close original NetCDF file.
    nc_fid.close()
    return nc_dump


def extract_vars_data(nc_file_path):
    """
    Extracts data of the variables from a .nc file

    Input: .nc file path

    Ouput: list of dict containing variables and dimensions data.
    Example, with 'air' as the variable and time/lat/lon as the dimensions:
    [
        {
            'time': 17628096.0,
            'lat': 90.0,
            'lon': 0.0,
            'air': 234.5
        }
    ]
    """
    # Dataset is the class behavior to open the file
    nc_fid = Dataset(nc_file_path, 'r')

    var_name_list = get_variables(nc_fid.variables, nc_fid.dimensions)

    if not are_vars_sharing_same_dims(nc_fid.variables, var_name_list):
        logger.error(
            'The variables do not share the same dimensions! Aborting...')
        return []

    # Extract data from NetCDF file
    dims_names_list = [
        x for x in nc_fid.variables[var_name_list[0]].dimensions]
    dims_size_list = [x for x in nc_fid.variables[var_name_list[0]].shape]

    logger.info('Let\'s go on a recursive adventure!')
    logger.info(f'{len(dims_size_list)} depths to explore')
    logger.info(
        'Note: the deepest level will not print output, so that the console is not overflowed')
    items = flatten_data(var_name_list,
                         nc_fid.variables, dims_names_list, dims_size_list)
    # Close original NetCDF file.
    nc_fid.close()

    logger.info('The recursive adventure is finished, hurray!')
    return items


def get_variables(nc_variables, nc_dims):
    """
    Extracts a list of variables from the list of variables + dimensions and the list of dimensions only

    Input: 
        nc_varibles: list of variables and dimensions
        nc_dims: list of dimensions only

    Output: list of variables only
    """
    # Keep only the variables
    vars_and_dims = [x for x in nc_variables]
    dims = [x for x in nc_dims]
    var_name_list = []
    for x in vars_and_dims:
        if x not in dims:
            # It's a variable!
            var_name_list.append(x)
    return var_name_list


def are_vars_sharing_same_dims(nc_variables, var_name_list):
    """
    Safety method to enforce that variables share the same dimensions (otherwise the rest of the extract_vars_data method would fail)

    Input:
        nc_variables: list of variables (including data)
        var_name_list: list of variables names (only names, no data or other metadata)

    Output:
        True: good to go, all variables share the same dimensions and shapes
        False: issue detected, extract_vars_data execution should abort
    """
    for i, var_name in enumerate(var_name_list):
        if i > 0:
            j = i
            while(j > 0):
                previous_var_name = var_name_list[j-1]
                same_dim = nc_variables[var_name].dimensions == nc_variables[previous_var_name].dimensions
                same_shape = nc_variables[var_name].shape == nc_variables[previous_var_name].shape
                if not same_dim or not same_shape:
                    logger.error(
                        f'The variables {var_name} and {previous_var_name} do not share the same dimensions and/or shapes. Aborting...')
                    logger.debug(
                        f'{var_name}: {nc_variables[var_name].dimensions}\n{nc_variables[var_name].shape}')
                    logger.debug(
                        f'{previous_var_name}: {nc_variables[previous_var_name].dimensions}\n{nc_variables[previous_var_name].shape}')
                    return False
                j -= 1
    return True


def flatten_data(var_name_list, nc_fid_variables, dims_names_list, dims_size_list, dims_data=[], vars_data=defaultdict(), items=[], depth=0):
    """
    Recursive method to extract the variables and dimensions data.
    The recursion is done on the dimensions, as their number is not known (depends on the .nc file)

    Input:
        var_name_list: list of the variable names
        nc_fid_variables: list of the variables (& dimensions) data
        dims_names_list: list of the dimension names (e.g. ['time', 'lat', 'lon'])
        dims_size_list: list of the dimension sizes (e.g. [4, 713, 366])
        dims_data=[]: recursion field to hold the dimension extracted data
        vars_data=defaultdict(): recursion field to hold the variables extracted data
        items=[]: recursion field to hold the final data (returned field)
        depth=0: recursion field, depth of the recursion

    Ouput: list of dict containing variables and dimensions data.
    Example, with 'air' as the variable and time/lat/lon as the dimensions:
    [
        {
            'time': 17628096.0,
            'lat': 90.0,
            'lon': 0.0,
            'air': 234.5
        }
    ]
    """
    stop = len(dims_size_list) - 1
    for i in range(0, dims_size_list[depth]):
        if depth == 0:
            logger.info(f'Depth {depth}: {i}/{dims_size_list[depth]}')
        elif depth < stop and i % 25 == 0:
            # The deepest level would output too many lines in the console
            tabs = depth * '\t'
            logger.info(f'Depth {depth}: {tabs}{i}/{dims_size_list[depth]}')
        dims_data.append(
            float(nc_fid_variables[dims_names_list[depth]][i].data))
        if depth == 0:
            # Initial values for the variables
            for var_name in var_name_list:
                vars_data[var_name] = [
                    {'data': nc_fid_variables[var_name][i]}]
            items = flatten_data(var_name_list,
                                 nc_fid_variables, dims_names_list, dims_size_list, dims_data, vars_data, items, depth+1)
        elif depth == stop:
            # Fill in the variables values
            item = defaultdict()
            for var_name in var_name_list:
                previous_value = vars_data[var_name][depth-1]['data']
                item[var_name] = float(previous_value[i])
            # Fill in the dimensions values
            for j, dim_name in enumerate(dims_names_list):
                item[dim_name] = dims_data[j]
            items.append(item)
        else:
            # Store values at this depth then dig deeper
            for var_name in var_name_list:
                previous_value = vars_data[var_name][depth-1]['data']
                vars_data[var_name].append({'data': previous_value[i]})
            items = flatten_data(var_name_list,
                                 nc_fid_variables, dims_names_list, dims_size_list, dims_data, vars_data, items, depth+1)
    return items
