import { Ionicons } from '@expo/vector-icons';
import { useNotificationListQuery } from 'api/notification/query';
import { useState } from 'react';
import { TouchableOpacity, View } from 'react-native';

import NotificationModal from './NotificationModal';

export default function NotificationButton() {
  const [visible, setVisible] = useState(false);
  const { isSuccess, data, refetch } = useNotificationListQuery();

  const hasNotification = data?.data.notificationList.some((item) => !item.isRead);

  return (
    <>
      <TouchableOpacity onPress={() => setVisible(true)} className="relative p-2">
        <Ionicons name="notifications" size={26} color="black" />
        {hasNotification && (
          <View className="absolute right-1 top-1 h-2 w-2 rounded-full bg-red-500" />
        )}
      </TouchableOpacity>

      {isSuccess && (
        <NotificationModal
          visible={visible}
          onClose={() => setVisible(false)}
          notifications={data?.data.notificationList}
          refetch={refetch}
        />
      )}
    </>
  );
}
