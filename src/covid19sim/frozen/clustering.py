import dataclasses
import numpy as np
import typing

import covid19sim.frozen.message_utils as mu


@dataclasses.dataclass
class SimpleCluster:
    """A simple message cluster.

    The default implementation of the 'fit' functions for this base class will
    simply attempt to merge new messages and adjust the risk level of the cluster
    to the average of all messages it aggregates.
    """

    uid: np.uint8
    """Unique Identifier (UID) of the cluster."""

    risk_level: np.uint8
    """Quantified risk level of the cluster."""

    first_update_time: np.uint64
    """Cluster creation timestamp."""

    latest_update_time: np.uint64
    """Latest cluster update timestamp."""

    messages: typing.List[mu.EncounterMessage] = dataclasses.field(default_factory=list)
    """List of encounter messages aggregated into this cluster (in added order)."""
    # note: messages above might have been

    ##########################################
    # private variables (for debugging only!)

    _real_encounter_uids: typing.List[np.uint64] = dataclasses.field(default_factory=list)
    """Real Unique Identifiers (UIDs) of the clustered user(s)."""

    _real_encounter_times: typing.List[np.uint64] = dataclasses.field(default_factory=list)
    """Real timestamp of the clustered encounter(s)."""

    _unclustered_messages: typing.List[mu.GenericMessageType] = dataclasses.field(default_factory=list)
    """List of all messages (encounter+update) messages that were used to update this cluster."""

    def _is_homogenous(self) -> bool:
        """Returns whether this cluster is truly homogenous (i.e. tied to one user) or not."""
        return len(np.unique([m._sender_uid for m in self._unclustered_messages])) <= 1

    @staticmethod
    def create_cluster_from_message(message: mu.GenericMessageType) -> "SimpleCluster":
        """Creates and returns a new cluster based on a single encounter message."""
        return SimpleCluster(
            # app-visible stuff below
            uid=message.uid,
            risk_level=message.risk_level
                if isinstance(message, mu.EncounterMessage) else message.new_risk_level,
            first_update_time=message.encounter_time,
            latest_update_time=message.encounter_time
                if isinstance(message, mu.EncounterMessage) else message.update_time,
            messages=[message] if isinstance(message, mu.EncounterMessage)
                else mu.create_encounter_from_update_message(message),
            # debug-only stuff below
            _real_encounter_uids=[message._sender_uid],
            _real_encounter_times=[message._real_encounter_time],
            _unclustered_messages=[message],  # once added, messages here should never be removed
        )

    def fit_encounter_message(self, message: mu.EncounterMessage):
        """Updates the current cluster given a new encounter message."""
        # note: this simplistic implementation will throw if UIDs dont perfectly match
        # the added message will automatically be used to adjust the cluster's risk level
        assert message.uid == self.uid, "cluster and new encounter message UIDs mismatch"
        self.latest_update_time = max(message.encounter_time, self.latest_update_time)
        self.messages.append(message)  # in this list, encounters may get updated
        self.risk_level = np.uint8(np.round(np.mean([m.risk_level for m in self.messages])))
        self._real_encounter_uids.append(message._sender_uid)
        self._real_encounter_times.append(message._real_encounter_time)
        self._unclustered_messages.append(message)  # in this list, messages NEVER get updated

    def fit_update_message(self, update_message: mu.UpdateMessage, adopt_if_orphan: bool = False):
        """Updates a message in the current cluster given a new update message."""
        # note: this simplistic implementation will throw if UIDs dont perfectly match
        found_match = None
        assert update_message.uid == self.uid, "cluster and new update message UIDs mismatch"
        # TODO: should we assert that all update messages are received in order based on update time?
        #       if the assumption is false, what should we do? currently, we try to apply anyway?
        # TODO: see if check below still valid when update messages are no longer systematically sent
        for encounter_message_idx, encounter_message in enumerate(self.messages):
            if encounter_message.risk_level == update_message.old_risk_level and \
                    encounter_message.uid == update_message.uid and \
                    encounter_message.encounter_time == update_message.encounter_time:
                found_match = (encounter_message_idx, encounter_message)
                break
        if found_match is not None:
            self.messages[found_match[0]] = mu.create_updated_encounter_with_message(
                encounter_message=found_match[1], update_message=update_message,
            )
        else:
            assert adopt_if_orphan, f"cannot adopt orphan update message: {update_message}"
            self.messages.append(mu.create_encounter_from_update_message(update_message))
        # note: the 'cluster update time' is still encounter-message-based, not update-message-based
        self.latest_update_time = max(update_message.encounter_time, self.latest_update_time)
        self.risk_level = np.uint8(np.round(np.mean([m.risk_level for m in self.messages])))
        self._real_encounter_uids.append(update_message._sender_uid)
        self._real_encounter_times.append(update_message._real_encounter_time)
        self._unclustered_messages.append(update_message)  # in this list, messages NEVER get updated

    def get_cluster_embedding(self) -> np.ndarray:
        """Returns the 'embeddings' array for this particular cluster."""
        # note: this returns an array of four 'features', i.e. the cluster UID, the cluster's
        #       average encounter risk level, the number of messages in the cluster, and
        #       the first encounter timestamp of the cluster. This array's type will be returned
        #       as np.uint64 to insure that no data is lost w.r.t. message counts or timestamps.
        return np.asarray([self.uid, self.risk_level,
                           len(self.messages), self.first_update_time], dtype=np.uint64)

    def _get_cluster_exposition_flag(self) -> bool:
        """Returns whether this particular cluster contains an exposition encounter."""
        # note: an 'exposition encounter' is an encounter where the user was exposed to the virus;
        #       this knowledge is UNOBSERVED (hence the underscore prefix in the function name), and
        #       relies on the flag being properly defined in the clustered messages
        return any([bool(m._exposition_event) for m in self.messages])


class SimplisticClusterManager:
    """Manages message cluster creation and updates.

    This class implement a simplistic clustering strategy where messages are only combined
    on a timestamp-level basis, meaning clusters cannot contain messages with different
    timestamps. The update messages can also never split a cluster into different parts.
    """

    clusters: typing.List[SimpleCluster]
    max_history_offset: int
    latest_refresh_timestamp: np.uint64
    add_orphan_updates_as_clusters: bool

    def __init__(
            self,
            max_history_offset: int,  # TODO: add default value in days? (24 * 60 * 60 * 14)?
            add_orphan_updates_as_clusters: bool = False,
            rng=np.random,
    ):
        self.clusters = []
        self.max_history_offset = max_history_offset
        self.latest_refresh_timestamp = np.uint64(0)
        self.add_orphan_updates_as_clusters = add_orphan_updates_as_clusters
        self.rng = rng

    def cleanup_old_clusters(self, current_timestamp: np.uint64):
        """Gets rid of clusters that are too old given the current timestamp."""
        to_keep = []
        for cluster_idx, cluster in enumerate(self.clusters):
            update_offset = int(current_timestamp) - int(cluster.latest_update_time)
            if update_offset <= self.max_history_offset:
                to_keep.append(cluster)
        self.clusters = to_keep

    def _check_if_message_outdated(self, message: mu.GenericMessageType, cleanup: bool = True) -> bool:
        """Returns whether a message is outdated or not. Will also refresh the internal check timestamp."""
        self.latest_refresh_timestamp = max(message.encounter_time, self.latest_refresh_timestamp)
        outdated = False
        if self.latest_refresh_timestamp:
            min_offset = int(self.latest_refresh_timestamp) - int(message.encounter_time)
            if min_offset > self.max_history_offset:
                # there's no way this message is useful if we get here, since it's so old
                outdated = True
            if cleanup:
                self.cleanup_old_clusters(self.latest_refresh_timestamp)
        return outdated

    def add_messages(self, messages: typing.Iterable[mu.GenericMessageType], cleanup: bool = True):
        """Dispatches the provided messages to the correct internal 'add' function based on type."""
        for message in messages:
            if isinstance(message, mu.EncounterMessage):
                self._add_encounter_message(message, cleanup)
            elif isinstance(message, mu.UpdateMessage):
                self._add_update_message(message, cleanup)
            else:
                ValueError("unexpected message type")

    def _add_encounter_message(self, message: mu.EncounterMessage, cleanup: bool = True):
        """Fits an encounter message to an existing cluster or creates a new cluster to own it."""
        if self._check_if_message_outdated(message, cleanup):
            return
        # simplistic clustering = we are looking for an exact day/uid/risk level match, or we create a new cluster
        matched_clusters = []
        for cluster in self.clusters:
            if cluster.uid == message.uid and \
                    cluster.risk_level == message.risk_level and \
                    cluster.first_update_time == message.encounter_time:
                matched_clusters.append(cluster)
        if matched_clusters:
            # the number of matched clusters might be greater than one if update messages caused
            # a cluster signature to drift into another cluster's; we will randomly assign this
            # encounter to one of the two (this is the naive part)
            matched_cluster = self.rng.choice(matched_clusters)
            matched_cluster.fit_encounter_message(message)
        else:
            # create a new cluster for this encounter alone
            self.clusters.append(SimpleCluster.create_cluster_from_message(message))

    def _add_update_message(self, message: mu.UpdateMessage, cleanup: bool = True):
        """Fits an update message to an existing cluster."""
        if self._check_if_message_outdated(message, cleanup):
            return
        matched_clusters = []
        for cluster in self.clusters:
            if cluster.uid == message.uid and cluster.first_update_time == message.encounter_time:
                # found a potential match based on uid and encounter time; check for actual
                # encounters in the cluster with the target risk level to update...
                for encounter in cluster.messages:
                    if encounter.risk_level == message.old_risk_level:
                        matched_clusters.append(cluster)
                        # one matching encounter is sufficient, we can update that cluster
                        break
        if matched_clusters:
            # the number of matched clusters might be greater than one if update messages caused
            # a cluster signature to drift into another cluster's; we will randomly assign this
            # encounter to one of the two (this is the naive part)
            matched_cluster = self.rng.choice(matched_clusters)
            matched_cluster.fit_update_message(message)
        else:
            if self.add_orphan_updates_as_clusters:
                self.clusters.append(SimpleCluster.create_cluster_from_message(message))
            else:
                raise AssertionError("could not find any proper cluster match for update message")

    def get_embeddings_array(self) -> np.ndarray:
        """Returns the 'embeddings' array for all clusters managed by this object."""
        return np.asarray([c.get_cluster_embedding() for c in self.clusters], dtype=np.uint64)

    def _get_expositions_array(self) -> np.ndarray:
        """Returns the 'expositions' array for all clusters managed by this object."""
        return np.asarray([c._get_cluster_exposition_flag() for c in self.clusters], dtype=np.uint8)


# class NaiveClusterManager(SimplisticClusterManager):
#     """Manages message cluster creation and updates.
#
#     This class implement a naive clustering strategy where encounter messages can be combined
#     into a single cluster despite being received on different days and despite the uncertainty
#     caused by the rolling UIDs. Update messages can also split clusters when needed.
#     """
#
#     cluster_day_map: typing.Dict[np.uint64, typing.List[SimpleCluster]]
#     # above: timestamps to clusters mapping, where np.uint64 is an arbitrarily discretized timestamp
#     max_day_history_length: int
#     latest_refresh_timestamp: np.uint64
#     add_orphan_updates_as_clusters: bool
#
#     def _cleanup_old_clusters(self, current_timestamp):
#         """Gets rid of clusters that are too old given the current timestamp."""
#         for cluster_timestamp in self.cluster_map:
#             if mu.get_days_offset(cluster_timestamp, current_timestamp) > self.max_day_history_length:
#                 del self.cluster_map[cluster_timestamp]
#
#     def add_messages(self, messages: typing.Iterable[mu.GenericMessageType]):
#         """Dispatches the provided messages to the correct internal 'add' function based on type."""
#         for message in messages:
#             if isinstance(message, mu.EncounterMessage):
#                 self._add_encounter_message(message)
#             elif isinstance(message, mu.UpdateMessage):
#                 self._add_update_message(message)
#             else:
#                 ValueError("unexpected message type")
#
#     def _add_encounter_message(self, message: mu.EncounterMessage):
#         """Fits an encounter message to an existing cluster or creates a new cluster to own it."""
#         prev_latest_refresh_timestamp = self.latest_refresh_timestamp
#         self.latest_refresh_timestamp = max(message.encounter_time, self.latest_refresh_timestamp)
#         if prev_latest_refresh_timestamp:
#             min_days_offset = mu.get_days_offset(message.encounter_time, prev_latest_refresh_timestamp)
#             if min_days_offset > self.max_day_history_length:
#                 # there's no way this message is useful if we get here, since it's so old
#                 return
#         if message.encounter_time not in self.cluster_map:
#             self.cluster_map[message.encounter_time] = []
#         target_clusters = self.cluster_map[message.encounter_time]
#         matched_clusters = []
#         for cluster in target_clusters:
#             # we will assume that even if we find a perfect UID match on the target day,
#             # if the risk levels don't match, we won't add it in
#             # TODO: see if still valid when update messages are no longer systematically sent
#             if cluster.uid == message.uid and cluster.risk_level == message.risk_level:
#                 matched_clusters.append(cluster)
#         if matched_clusters:
#             # the number of matched clusters might be greater than one if update messages caused
#             # a cluster signature to drift into another cluster's; we will randomly assign this
#             # encounter to one of the two (this is the naive part)
#             matched_cluster = self.rng.choice(matched_clusters)
#             matched_cluster.fit_encounter_message(message)
#         else:
#             # create a new cluster for this encounter alone
#             target_clusters.append(NaiveCluster.create_cluster_from_encounter(message))
#
#     def _add_update_message(self, message: mu.UpdateMessage):
#         """Fits an update message to an existing cluster."""
#
#         if self.add_orphan_updates_as_clusters:
#             raise NotImplementedError  # TODO
#         # add_orphan_updates_as_clusters
#
#         # assert message.uid == self.uid  # can only update perfect matches!
#         # # TODO: see if check below still valid when update messages are no longer systematically sent
#         # assert message.old_risk_level == self.risk_level
#         # # TODO: check what risk is sent when multi updates for same encounter
#         # assert message.encounter_time == self.first_update_time
#         # # TODO: should we assert that all update messages are received in order based on update time?
#         # #       if the assumption is false, what should we do? currently, we try to apply anyway
#
#     def add_messages(self, messages, current_day, rng=None):
#         """ This function clusters new messages by scoring them against old messages in a sort of naive nearest neighbors approach"""
#         for message in messages:
#             m_dec = decode_message(message)
#             best_cluster, _, best_score = self.score_matches(m_dec, current_day, rng=rng)
#             self.num_messages += 1
#             self.clusters[best_cluster].append(message)
#             self.add_to_clusters_by_day(best_cluster, m_dec.day, message)
#
#     def score_matches(self, m_new, current_day, rng=None):
#         """ This function checks a new risk message against all previous messages, and assigns to the closest one in a brute force manner"""
#         best_score = 2
#         cluster_days = hash_to_cluster_day(m_new)
#         best_cluster = hash_to_cluster(m_new)
#
#         if self.clusters_by_day[current_day].get(best_cluster, None):
#             return (best_cluster, None, 3)
#         found = False
#         for day, cluster_ids in cluster_days.items():
#             for cluster_id in cluster_ids:
#                 if self.clusters_by_day[current_day - day].get(cluster_id, None):
#                     best_cluster = cluster_id
#                     found = True
#                     break
#             if found:
#                 break
#             best_score -= 1
#
#         return best_cluster, None, best_score
#
#     def update_records(self, update_messages):
#         # if we're using naive tracing, we actually don't care which records we update
#         if not update_messages:
#             return self
#
#         grouped_update_messages = self.group_by_received_at(update_messages)
#         for received_at, update_messages in grouped_update_messages.items():
#             old_cluster = None
#             for update_message in update_messages:
#                 old_message_dec = Message(update_message.uid, update_message.risk, update_message.day,
#                                           update_message.unobs_id, update_message.has_app)
#                 old_message_enc = encode_message(old_message_dec)
#                 updated_message = Message(old_message_dec.uid, update_message.new_risk, old_message_dec.day,
#                                           old_message_dec.unobs_id, old_message_dec.has_app)
#                 new_cluster = hash_to_cluster(updated_message)
#                 self.update_record(old_cluster, new_cluster, old_message_dec, updated_message)
#         return self
#
#     def update_record(self, old_cluster_id, new_cluster_id, message, updated_message):
#         """ This function updates a message in all of the data structures and can change the cluster that this message is in"""
#         old_m_enc = encode_message(message)
#         new_m_enc = encode_message(updated_message)
#
#         del self.clusters[old_cluster_id][self.clusters[old_cluster_id].index(old_m_enc)]
#         del self.clusters_by_day[message.day][old_cluster_id][
#             self.clusters_by_day[message.day][old_cluster_id].index(old_m_enc)]
#
#         self.clusters[new_cluster_id].append(encode_message(updated_message))
#         self.add_to_clusters_by_day(new_cluster_id, updated_message.day, new_m_enc)
#
#     def add_to_clusters_by_day(self, cluster, day, m_i_enc):
#         if self.clusters_by_day[day].get(cluster):
#             self.clusters_by_day[day][cluster].append(m_i_enc)
#         else:
#             self.clusters_by_day[day][cluster] = [m_i_enc]
#
#     def group_by_received_at(self, update_messages):
#         """ This function takes in a set of update messages received during some time interval and clusters them based on how near in time they were received"""
#         # TODO: We need more information about the actual implementation of the message protocol to use this.\
#         # TODO: it is possible that received_at is actually the same for all update messages under the protocol, in which case we can delete this function.
#         TIME_THRESHOLD = datetime.timedelta(minutes=1)
#         grouped_messages = defaultdict(list)
#         for m1 in update_messages:
#             m1 = decode_update_message(m1)
#             # if m1.received_at - received_at < TIME_THRESHOLD and -(m1.received_at - received_at) < TIME_THRESHOLD:
#             #     grouped_messages[received_at].append(m1)
#             # else:
#             grouped_messages[m1.received_at].append(m1)
#
#         return grouped_messages
#
#     def purge(self, current_day):
#         for cluster_id, messages in self.clusters_by_day[current_day - 14].items():
#             for message in messages:
#                 del self.clusters[cluster_id][self.clusters[cluster_id].index(message)]
#                 self.num_messages -= 1
#         to_purge = []
#         for cluster_id, messages in self.clusters.items():
#             if len(self.clusters[cluster_id]) == 0:
#                 to_purge.append(cluster_id)
#         for cluster_id in to_purge:
#             del self.clusters[cluster_id]
#         if current_day - 14 >= 0:
#             del self.clusters_by_day[current_day - 14]
#         to_purge = defaultdict(list)
#         for day, clusters in self.clusters_by_day.items():
#             for cluster_id, messages in clusters.items():
#                 if not messages:
#                     to_purge[day].append(cluster_id)
#         for day, cluster_ids in to_purge.items():
#             for cluster_id in cluster_ids:
#                 del self.clusters_by_day[day][cluster_id]
#         self.update_messages = []
#
#     def __len__(self):
#         return self.num_messages
