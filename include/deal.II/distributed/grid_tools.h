// ---------------------------------------------------------------------
//
// Copyright (C) 2017 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

#ifndef dealii_distributed_grid_tools_h
#define dealii_distributed_grid_tools_h


#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/distributed/tria_base.h>

DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include <boost/optional.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>

#ifdef DEAL_II_WITH_ZLIB
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#endif

DEAL_II_ENABLE_EXTRA_DIAGNOSTICS

#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace parallel
{

  /**
   * This namespace defines parallel algorithms that operate on meshes.
   */
  namespace GridTools
  {
#ifdef DEAL_II_WITH_MPI
    /**
     * Distributed compute point locations: similarly to
     * GridTools::compute_point_locations given a @p cache and a list of
     * @p local_points create the quadrature rules for a distributed triangulation.
     *
     * @param[in] cache a GridTools::Cache object
     * @param[in] points the array of points owned by the current process. Every
     *  process can have a different array of points which can be empty and not
     *  contained within the locally owned part of the triangulation.
     * @param[out] tuple containing the quadrature information
     *
     * The elements of the output tuple are:
     * - cells : a vector of a vector cells of the all cells containing at
     *  least a point.
     * - qpoints : a vector of vector of points. Each entry contains the locally owned
     *  points transformed in the unit cell.
     * - maps : a vector indices of vector of integers, containing the mapping between
     *  local numbering in qpoints, and returned points.
     * - points : a vector of points contained inside the locally owned part of
     *  the mesh.
     * - owners : a vector containing the rank of the process owning the corresponding
     *  element of points.
     *
     * @author Giovanni Alzetta, 2017
     */
    template <int dim, int spacedim>
    std::tuple<
        std::vector< typename Triangulation<dim, spacedim>::active_cell_iterator >,
        std::vector< std::vector< Point<dim> > >,
        std::vector< std::vector<unsigned int> >,
        std::vector< std::vector< Point<spacedim> > >,
        std::vector< unsigned int >
        >
        distributed_compute_point_locations
                                (const Cache<dim,spacedim>                &cache,
                                 const std::vector<Point<spacedim> >      &local_points,
                                 MPI_Comm                              mpi_communicator)
        // Developing it here for now, to avoid conflicts on many diffrerent files...
    {
      // Step 1: global description of the mesh using bounding boxes TO DO CACHE PART
      std::vector< std::vector< BoundingBox<spacedim> > >
          global_bboxes = cache.get_global_bboxes(); // this shall get the bounding boxes describing the whole space

      // Step 2: Using the bounding boxes to guess the
      // owner of each  point in local_points
      unsigned int n_procs = Utilities::MPI::n_mpi_processes(mpi_communicator);
      unsigned int proc = Utilities::MPI::this_mpi_process(mpi_communicator);

      std::vector< std::vector< Point<spacedim> > point_owners(n_procs);
      for(const Point<spacedim> & pt: local_points)
      {
        bool possibly_local = false;
        // Check if the point is in the current part of the mesh
        for(const auto & bbox: global_bboxes[proc])
          if(bbox->point_inside(pt))
          {
            point_owners[proc].push_back(pt);
            possibly_local = true;
            break;
          }

        if(possibly_local)
          continue;
        else
          for(unsigned int rk=0;rk<n_procs;++rk)
            if(rk!=proc)
              for(const auto & bbox: global_bboxes[rk])
                if(bbox->point_inside(pt))
                {
                  point_owners[rk].push_back(pt);
                  break;
                }
      }

      // Step 3: Use GridTools::compute_point_locatios to study local points
      // and handle points found in artificial/ghost cells
      auto local_cell_qpoint_map = GridTools::compute_point_locations(cache,point_owners[proc]);
      for(unsigned int c=0; c< std::get<0>(local_cell_qpoint_map).size(); ++c)
      {
        if( std::get<0>(local_cell_qpoint_map)[c]->is_artificial())
          for(unsigned int i=0; i<std::get<1>(local_cell_qpoint_map)[c].size(); ++i)
            for(unsigned int rk=0;rk<n_procs;++rk)
                if(rk!=proc)
                  for(const auto & bbox: global_bboxes[rk])
                    if(bbox->point_inside(local_points(std::get<2>(local_cell_qpoint_map)[c][i])))
                    {
                      point_owners[rk].push_back
                          (local_points(std::get<2>(local_cell_qpoint_map)[c][i]));
                      break;
                    }
        // Points in ghost cells are correctly indentified by their subdomain id
        else if( std::get<0>(local_cell_qpoint_map)[c]->is_ghost() )
        {
          unsigned int rk = std::get<0>(local_cell_qpoint_map)[c]->subdomain_id();
          for(unsigned int i=0; i<std::get<1>(local_cell_qpoint_map)[c].size(); ++i)
              point_owners[rk].push_back
                  (local_points(std::get<2>(local_cell_qpoint_map)[c][i]));
          // Actually the transfomed qpoint is correct, but this simpler to handle
          // I don't think transferring the data is much faster than re computing
          // in loco...this is a good question, I guess
        }
      }

      // Step 4: sending/receiving points
      // Begin by communicating the number of points to be sent out
      std::vector< unsigned int > send_amount(n_procs,0);
      // How many points need to be communicated with each process ?
      for(unsigned int rk=0; rk< n_procs; ++rk)
        if(rk!= proc)
        send_amount[rk] = point_owners[rk].size();

      std::vector< unsigned int > receive_amount(n_procs);

      MPI_Alltoall(&send_amount[0], 1, MPI_UNSIGNED,
                   &receive_amount[0], 1, MPI_UNSIGNED,
                   mpi_communicator);

      // Sending/receiving part
      unsigned int count = sizeof(Point<spacedim>)/sizeof(double);
      static_assert(sizeof(Point<spacedim>)==count*sizeof(double),
                    "Error in point type creation: size not matching");
      MPI_Datatype ptype;
      MPI_Type_contiguous(count,MPI_DOUBLE,&ptype);
      MPI_Type_commit(&ptype);

      unsigned int sum_received = 0;

      std::vector< Point<spacedim> > received_points;
      std::vector< unsigned int > received_rank;

      // One to one communications between processes
      for(unsigned int rk=0; rk < n_procs; ++rk)
      {
        if(rk != proc)
        {
          // Sending part
          if(send_amount[rk] != 0)
          {
            MPI_Request request;
            MPI_Isend(&(point_owners[rk][0]), send_amount[rk], ptype,
                     rk, (1+rk)*proc, mpi_communicator, &request);
          }
          // Receiving part
          if(receive_amount[rk] != 0)
          {
            MPI_Request request;
            received_points.resize(sum_received + receive_amount[rk]);
            MPI_Irecv(&(received_points[sum_received]), receive_amount[rk], ptype,
                     proc, (1+proc)*rk, mpi_communicator, &request);
            sum_received += receive_amount[rk];
            std::vector<unsigned int> rks(receive_amount[rk],rk);
            received_rank.insert(received_rank.end(), rks.begin(), rks.end());
          }
        }
      }

      // Step 5: running compute point locations on the received points
      // and creating output
      std::tuple<
          std::vector< typename Triangulation<dim, spacedim>::active_cell_iterator >,
          std::vector< std::vector< Point<dim> > >,
          std::vector< std::vector<unsigned int> >,
          std::vector< std::vector< Point<spacedim> > >,
          std::vector< unsigned int >
          > out_cell_qpt_map_pt_rk; // output vector

      std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator >,
          std::vector< std::vector< Point<dim> > >,
          std::vector< std::vector<unsigned int> > > other_cell_qpoint_map =
            GridTools::compute_point_locations(cache,received_points);

      // Now prepare output...its actually a pain to mix local_compute_point_locations and the new one...

      return out_cell_qpt_map_pt_rk;
    }
#endif // DEAL_II_WITH_MPI .,..actually should move it inside and write a nonmpi version
       // which is a sort of a wrapper for compute_pt_loc (and put #endif in better place)

    /**
     * Exchange arbitrary data of type @p DataType provided by the function
     * objects from locally owned cells to ghost cells on other processors.
     *
     * After this call, you typically will have received data from @p unpack on
     * every ghost cell as it was given by @p pack on the owning processor.
     * Whether you do or do not receive information to @p unpack on a given
     * ghost cell depends on whether the @p pack function decided that
     * something needs to be sent. It does so using the boost::optional
     * mechanism: if the boost::optional return object of the @p pack
     * function is empty, then this implies that no data has to be sent for
     * the locally owned cell it was called on. In that case, @p unpack will
     * also not be called on the ghost cell that corresponds to it on the
     * receiving side. On the other hand, if the boost::optional object is
     * not empty, then the data stored within it will be sent to the received
     * and the @p unpack function called with it.
     *
     * @tparam DataType The type of the data to be communicated. It is assumed
     *   to be serializable by boost::serialization. In many cases, this
     *   data type can not be deduced by the compiler, e.g., if you provide
     *   lambda functions for the second and third argument
     *   to this function. In this case, you have to explicitly specify
     *   the @p DataType as a template argument to the function call.
     * @tparam MeshType The type of @p mesh.
     *
     * @param mesh A variable of a type that satisfies the requirements of the
     * @ref ConceptMeshType "MeshType concept".
     * @param pack The function that will be called on each locally owned cell
     *   that is a ghost cell somewhere else. As mentioned above, the function
     *   may return a regular data object of type @p DataType to indicate
     *   that data should be sent, or an empty
     *   <code>boost::optional@<DataType@></code> to indicate that nothing has
     *   to be sent for this cell.
     * @param unpack The function that will be called for each ghost cell
     *   for which data was sent, i.e., for which the @p pack function
     *   on the sending side returned a non-empty boost::optional object.
     *   The @p unpack function is then called with the data sent by the
     *   processor that owns that cell.
     *
     *
     * <h4> An example </h4>
     *
     * Here is an example that shows how this function is to be used
     * in a concrete context. It is taken from the code that makes
     * sure that the @p active_fe_index (a single unsigned integer) is
     * transported from locally owned cells where one can set it in
     * hp::DoFHandler objects, to the corresponding ghost cells on
     * other processors to ensure that one can query the right value
     * also on those processors:
     * @code
     *    auto pack
     *    = [] (const typename dealii::hp::DoFHandler<dim,spacedim>::active_cell_iterator &cell) -> unsigned int
     *    {
     *      return cell->active_fe_index();
     *    };
     *
     *    auto unpack
     *      = [] (const typename dealii::hp::DoFHandler<dim,spacedim>::active_cell_iterator &cell,
     *            const unsigned int                                                        &active_fe_index) -> void
     *    {
     *      cell->set_active_fe_index(active_fe_index);
     *    };
     *
     *    parallel::GridTools::exchange_cell_data_to_ghosts<unsigned int, dealii::hp::DoFHandler<dim,spacedim>>
     *        (dof_handler, pack, unpack);
     * @endcode
     *
     * You will notice that the @p pack lambda function returns an `unsigned int`,
     * not a `boost::optional<unsigned int>`. The former converts automatically
     * to the latter, implying that data will always be transported to the
     * other processor.
     *
     * (In reality, the @p unpack function needs to be a bit more
     * complicated because it is not allowed to call
     * DoFAccessor::set_active_fe_index() on ghost cells. Rather, the
     * @p unpack function directly accesses internal data structures. But
     * you get the idea -- the code could, just as well, have exchanged
     * material ids, user indices, boundary indictors, or any kind of other
     * data with similar calls as the ones above.)
     */
    template <typename DataType, typename MeshType>
    void
    exchange_cell_data_to_ghosts (const MeshType &mesh,
                                  const std::function<boost::optional<DataType> (const typename MeshType::active_cell_iterator &)> &pack,
                                  const std::function<void (const typename MeshType::active_cell_iterator &, const DataType &)> &unpack);

    /**
     * A structure that allows the transfer of cell data of type @p T from one processor
     * to another. It corresponds to a packed buffer that stores a vector of
     * CellId and a vector of type @p T.
     *
     * This class facilitates the transfer by providing the save/load functions
     * that are able to pack up the vector of CellId's and the associated
     * data of type @p T into a stream.
     *
     * Type @p T is assumed to be serializable by <code>boost::serialization</code> (for
     * example <code>unsigned int</code> or <code>std::vector@<double@></code>).
     */
    template <int dim, typename T>
    struct CellDataTransferBuffer
    {
      /**
       * A vector to store IDs of cells to be transfered.
       */
      std::vector<CellId> cell_ids;

      /**
       * A vector of cell data to be transfered.
       */
      std::vector<T> data;

      /**
       * Write the data of this object to a stream for the purpose of
       * serialization.
       *
       * @pre The user is responsible to keep the size of @p data
       * equal to the size as @p cell_ids .
       */
      template <class Archive>
      void save (Archive &ar,
                 const unsigned int version) const;

      /**
       * Read the data of this object from a stream for the purpose of
       * serialization. Throw away the previous content.
       */
      template <class Archive>
      void load (Archive &ar,
                 const unsigned int version);

      BOOST_SERIALIZATION_SPLIT_MEMBER()

      /**
       * Pack the data that corresponds to this object into a buffer in
       * the form of a vector of chars and return it.
       */
      std::vector<char> pack_data () const;

      /**
       * Given a buffer in the form of an array of chars, unpack it and
       * restore the current object to the state that it was when
       * it was packed into said buffer by the pack_data() function.
       */
      void unpack_data (const std::vector<char> &buffer);

    };

  }


}

#ifndef DOXYGEN

namespace parallel
{
  namespace GridTools
  {

    template <int dim, typename T>
    template <class Archive>
    void
    CellDataTransferBuffer<dim,T>::save (Archive &ar,
                                         const unsigned int /*version*/) const
    {
      Assert(cell_ids.size() == data.size(),
             ExcDimensionMismatch(cell_ids.size(), data.size()));
      // archive the cellids in an efficient binary format
      const size_t n_cells = cell_ids.size();
      ar &n_cells;
      for (auto &it : cell_ids)
        {
          CellId::binary_type binary_cell_id = it.template to_binary<dim>();
          ar &binary_cell_id;
        }

      ar &data;
    }



    template <int dim, typename T>
    template <class Archive>
    void
    CellDataTransferBuffer<dim,T>::load (Archive &ar,
                                         const unsigned int /*version*/)
    {
      size_t n_cells;
      ar &n_cells;
      cell_ids.clear();
      cell_ids.reserve(n_cells);
      for (unsigned int c=0; c<n_cells; ++c)
        {
          CellId::binary_type value;
          ar &value;
          cell_ids.emplace_back(std::move(value));
        }
      ar &data;
    }



    template <int dim, typename T>
    std::vector<char>
    CellDataTransferBuffer<dim,T>::pack_data () const
    {
      // set up a buffer and then use it as the target of a compressing
      // stream into which we serialize the current object
      std::vector<char> buffer;
      {
#ifdef DEAL_II_WITH_ZLIB
        boost::iostreams::filtering_ostream out;
        out.push(boost::iostreams::gzip_compressor
                 (boost::iostreams::gzip_params
                  (boost::iostreams::gzip::best_compression)));
        out.push(boost::iostreams::back_inserter(buffer));

        boost::archive::binary_oarchive archive(out);
        archive << *this;
        out.flush();
#else
        std::ostringstream out;
        boost::archive::binary_oarchive archive(out);
        archive << *this;
        const std::string &s = out.str();
        buffer.reserve(s.size());
        buffer.assign(s.begin(), s.end());
#endif
      }

      return buffer;
    }


    template <int dim, typename T>
    void
    CellDataTransferBuffer<dim,T>::unpack_data (const std::vector<char> &buffer)
    {
      std::string decompressed_buffer;

      // first decompress the buffer
      {
#ifdef DEAL_II_WITH_ZLIB
        boost::iostreams::filtering_ostream decompressing_stream;
        decompressing_stream.push(boost::iostreams::gzip_decompressor());
        decompressing_stream.push(boost::iostreams::back_inserter(decompressed_buffer));
        decompressing_stream.write (buffer.data(), buffer.size());
#else
        decompressed_buffer.assign (buffer.begin(), buffer.end());
#endif
      }

      // then restore the object from the buffer
      std::istringstream in(decompressed_buffer);
      boost::archive::binary_iarchive archive(in);

      archive >> *this;
    }



    template <typename DataType, typename MeshType>
    void
    exchange_cell_data_to_ghosts (const MeshType &mesh,
                                  const std::function<boost::optional<DataType> (const typename MeshType::active_cell_iterator &)> &pack,
                                  const std::function<void (const typename MeshType::active_cell_iterator &, const DataType &)> &unpack)
    {
#ifndef DEAL_II_WITH_MPI
      (void)mesh;
      (void)pack;
      (void)unpack;
      Assert(false, ExcMessage("parallel::GridTools::exchange_cell_data_to_ghosts() requires MPI."));
#else
      constexpr int dim = MeshType::dimension;
      constexpr int spacedim = MeshType::space_dimension;
      auto tria =
        static_cast<const parallel::Triangulation<dim, spacedim>*>(&mesh.get_triangulation());
      Assert (tria != nullptr,
              ExcMessage("The function exchange_cell_data_to_ghosts() only works with parallel triangulations."));

      // map neighbor_id -> data_buffer where we accumulate the data to send
      typedef std::map<dealii::types::subdomain_id, CellDataTransferBuffer<dim, DataType> >
      DestinationToBufferMap;
      DestinationToBufferMap destination_to_data_buffer_map;

      std::map<unsigned int, std::set<dealii::types::subdomain_id> >
      vertices_with_ghost_neighbors = tria->compute_vertices_with_ghost_neighbors();

      for (auto cell : tria->active_cell_iterators())
        if (cell->is_locally_owned())
          {
            std::set<dealii::types::subdomain_id> send_to;
            for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
              {
                const std::map<unsigned int, std::set<dealii::types::subdomain_id> >::const_iterator
                neighbor_subdomains_of_vertex
                  = vertices_with_ghost_neighbors.find (cell->vertex_index(v));

                if (neighbor_subdomains_of_vertex ==
                    vertices_with_ghost_neighbors.end())
                  continue;

                Assert(neighbor_subdomains_of_vertex->second.size()!=0,
                       ExcInternalError());

                send_to.insert(neighbor_subdomains_of_vertex->second.begin(),
                               neighbor_subdomains_of_vertex->second.end());
              }

            if (send_to.size() > 0)
              {
                // this cell's data needs to be sent to someone
                typename MeshType::active_cell_iterator
                mesh_it (tria, cell->level(), cell->index(), &mesh);

                const boost::optional<DataType> data = pack(mesh_it);

                if (data)
                  {
                    const CellId cellid = cell->id();

                    for (auto it : send_to)
                      {
                        const dealii::types::subdomain_id subdomain = it;

                        // find the data buffer for proc "subdomain" if it exists
                        // or create an empty one otherwise
                        typename DestinationToBufferMap::iterator p
                          = destination_to_data_buffer_map.insert (std::make_pair(subdomain,
                                                                                  CellDataTransferBuffer<dim, DataType>()))
                            .first;

                        p->second.cell_ids.emplace_back(cellid);
                        p->second.data.emplace_back(data.get());
                      }
                  }
              }
          }


      // 2. send our messages
      std::set<dealii::types::subdomain_id> ghost_owners = tria->ghost_owners();
      const unsigned int n_ghost_owners = ghost_owners.size();
      std::vector<std::vector<char> > sendbuffers (n_ghost_owners);
      std::vector<MPI_Request> requests (n_ghost_owners);

      unsigned int idx=0;
      for (auto it = ghost_owners.begin();
           it!=ghost_owners.end();
           ++it, ++idx)
        {
          CellDataTransferBuffer<dim, DataType> &data = destination_to_data_buffer_map[*it];

          // pack all the data into the buffer for this recipient and send it.
          // keep data around till we can make sure that the packet has been
          // received
          sendbuffers[idx] = data.pack_data ();
          const int ierr = MPI_Isend(sendbuffers[idx].data(), sendbuffers[idx].size(),
                                     MPI_BYTE, *it,
                                     786, tria->get_communicator(), &requests[idx]);
          AssertThrowMPI(ierr);
        }

      // 3. receive messages
      std::vector<char> receive;
      for (unsigned int idx=0; idx<n_ghost_owners; ++idx)
        {
          MPI_Status status;
          int len;
          int ierr = MPI_Probe(MPI_ANY_SOURCE, 786, tria->get_communicator(), &status);
          AssertThrowMPI(ierr);
          ierr = MPI_Get_count(&status, MPI_BYTE, &len);
          AssertThrowMPI(ierr);

          receive.resize(len);

          char *ptr = receive.data();
          ierr = MPI_Recv(ptr, len, MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG,
                          tria->get_communicator(), &status);
          AssertThrowMPI(ierr);

          CellDataTransferBuffer<dim, DataType> cellinfo;
          cellinfo.unpack_data(receive);

          DataType *data = cellinfo.data.data();
          for (unsigned int c=0; c<cellinfo.cell_ids.size(); ++c, ++data)
            {
              const typename Triangulation<dim,spacedim>::cell_iterator
              tria_cell = cellinfo.cell_ids[c].to_cell(*tria);

              const typename MeshType::active_cell_iterator
              cell (tria, tria_cell->level(), tria_cell->index(), &mesh);

              unpack(cell, *data);
            }
        }

      // make sure that all communication is finished
      // when we leave this function.
      if (requests.size())
        {
          const int ierr = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
          AssertThrowMPI(ierr);
        }
#endif // DEAL_II_WITH_MPI
    }

  }
}

#endif // DOXYGEN

DEAL_II_NAMESPACE_CLOSE

#endif
