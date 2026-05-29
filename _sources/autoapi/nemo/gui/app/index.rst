nemo.gui.app
============

.. py:module:: nemo.gui.app


Classes
-------

.. autoapisummary::

   nemo.gui.app.NemoGUI


Functions
---------

.. autoapisummary::

   nemo.gui.app.launch


Module Contents
---------------

.. py:class:: NemoGUI

   Bases: :py:obj:`tkinter.Tk`


   Toplevel widget of Tk which represents mostly the main window
   of an application. It has an associated Tcl interpreter.


   .. py:attribute:: N_CARDS
      :value: 4



   .. py:attribute:: GIF_INTERVAL_MS
      :value: 220



   .. py:attribute:: cards
      :type:  list[nemo.gui.card.CubeCard]
      :value: []



   .. py:attribute:: BANNER_LINKS_TOP_FRAC
      :value: 0.24



   .. py:method:: current_gif_channel()


   .. py:method:: refresh_gif_clock()


.. py:function:: launch()

