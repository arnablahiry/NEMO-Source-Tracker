nemo.gui.card
=============

.. py:module:: nemo.gui.card


Classes
-------

.. autoapisummary::

   nemo.gui.card.CubeCard


Module Contents
---------------

.. py:class:: CubeCard(master, index: int, name: str, description: str, app=None, on_loaded=None, **kw)

   Bases: :py:obj:`tkinter.Frame`


   Frame widget which may contain other widgets and can have a 3D border.


   .. py:attribute:: index


   .. py:attribute:: name


   .. py:attribute:: description


   .. py:attribute:: enabled


   .. py:attribute:: cube_raw
      :value: None



   .. py:attribute:: cube
      :value: None



   .. py:attribute:: vel_array
      :value: None



   .. py:attribute:: scaling


   .. py:attribute:: filepath
      :value: None



   .. py:attribute:: beam
      :value: None



   .. py:attribute:: pixscale
      :value: None



   .. py:attribute:: detections
      :value: None



   .. py:attribute:: flow_seq
      :value: None



   .. py:method:: show_gif_for_channel(ch: int)


   .. py:method:: enable()


   .. py:method:: reset()

      Restore card to its initial state, clearing all results and logs.

      Card 0 preserves its loaded cube; cards 1+ wipe everything.



